# monitor/graph_delegate_monitor.py
# Enhanced App-only Graph monitor with PostgreSQL-backed admin settings & caution words
# Optimized for high-volume processing with pagination and batch operations

import os
import time
import requests
import re
from datetime import datetime, timedelta
from msal import ConfidentialClientApplication
from db_config import get_db_connection, ensure_schema, init_connection_pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# ---------------- CONFIG from ENV (defaults; DB can override these each cycle) ----------------
CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
DEBUG_CC = os.getenv("DEBUG_CC", "False").lower() in ("1", "true", "yes")
MAILBOX_LIST = os.getenv("MAILBOX_LIST", "")
MODEL_DIR = os.getenv("INFERENCE_MODEL_DIR", "/data/model")
FETCH_READ_EMAILS = os.getenv("FETCH_READ_EMAILS", "False").lower() in ("1", "true", "yes")
MAX_EMAILS_PER_POLL = int(os.getenv("MAX_EMAILS_PER_POLL", "50"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "7"))
MARK_AS_READ_ENV = os.getenv("MARK_AS_READ", "False").lower() in ("1", "true", "yes")
NEG_PROB_THRESH = float(os.getenv("NEG_PROB_THRESH", "0.06"))

# ---------------------------------------------------------------------------------------------
if not CLIENT_ID or not TENANT_ID or not CLIENT_SECRET:
    raise SystemExit("❌ CLIENT_ID, TENANT_ID and CLIENT_SECRET must be set.")

GRAPH_V1 = "https://graph.microsoft.com/v1.0"

# Try to import local classifier
try:
    from inference_local import classify_email
except Exception as e:
    print(f"⚠️ Warning: inference_local.classify_email not found; using fallback. {e}")
    def classify_email(text, model_dir=None, apply_rule=True, neg_prob_thresh=0.06, caution_keywords=None):
        """Fallback classifier when inference_local is not available"""
        t = (text or "").lower()
        # Check caution keywords first (if provided)
        if caution_keywords:
            for kw in caution_keywords:
                if kw and kw.lower() in t:
                    return {
                        "final_label": "Negative",
                        "pred_label": "Negative",
                        "probs": [0.9, 0.08, 0.02],
                        "postprocess_reason": f"caution_keyword:{kw}"
                    }
        # Simple keyword-based classification
        if any(x in t for x in ("disappoint", "unresolved", "delay", "not happy", "unable", "concern", "issue", "problem", "urgent", "complaint", "frustrated","gaps", "missing", "bad", "angry", "dissatisfied")):
            return {"final_label": "Negative", "pred_label": "Negative", "probs": [0.9, 0.08, 0.02], "postprocess_reason": "fallback"}
        if any(x in t for x in ("thank", "great", "well done", "congrat", "excellent", "appreciate", "happy", "pleased")):
            return {"final_label": "Positive", "pred_label": "Positive", "probs": [0.01, 0.05, 0.94], "postprocess_reason": "fallback"}
        return {"final_label": "Neutral", "pred_label": "Neutral", "probs": [0.05, 0.9, 0.05], "postprocess_reason": "fallback"}

# ---------------- Stats tracking ----------------
stats = {
    'total_processed': 0,
    'by_mailbox': {},
    'by_sentiment': {'Negative': 0, 'Neutral': 0, 'Positive': 0},
    'errors': 0,
    'start_time': datetime.utcnow()
}

def update_stats(mailbox, sentiment):
    """Update processing statistics"""
    stats['total_processed'] += 1
    stats['by_mailbox'][mailbox] = stats['by_mailbox'].get(mailbox, 0) + 1
    stats['by_sentiment'][sentiment] = stats['by_sentiment'].get(sentiment, 0) + 1

def print_stats():
    """Print current statistics"""
    runtime = datetime.utcnow() - stats['start_time']
    print("\n" + "="*60)
    print("📊 PROCESSING STATISTICS")
    print("="*60)
    print(f"⏱️ Runtime: {runtime}")
    print(f"📧 Total Processed: {stats['total_processed']}")
    print(f"❌ Errors: {stats['errors']}")
    print("\nBy Mailbox:")
    for mailbox, count in sorted(stats['by_mailbox'].items()):
        print(f"  📬 {mailbox}: {count}")
    print("\nBy Sentiment:")
    for sentiment, count in stats['by_sentiment'].items():
        emoji = {"Negative": "🔴", "Neutral": "🟡", "Positive": "🟢"}.get(sentiment, "⚪")
        print(f"  {emoji} {sentiment}: {count}")
    print("="*60 + "\n")

# ---------------- DB helpers for admin settings & caution words ----------------
def ensure_admin_tables():
    """Create admin_settings and caution_words if not present."""
    SQL = """
    CREATE TABLE IF NOT EXISTS caution_words (
        id SERIAL PRIMARY KEY,
        word TEXT NOT NULL UNIQUE,
        created_at TIMESTAMPTZ DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS admin_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TIMESTAMPTZ DEFAULT now()
    );
    -- Seed default settings if not exist
    INSERT INTO admin_settings (key, value)
    VALUES
        ('neg_prob_thresh', '0.06'),
        ('poll_interval', '30'),
        ('max_emails_per_poll', '50'),
        ('lookback_days', '7'),
        ('fetch_read_emails', 'false'),
        ('mark_as_read', 'false')
    ON CONFLICT (key) DO NOTHING;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SQL)
                conn.commit()
        print("✅ Admin tables ensured")
    except Exception as e:
        print(f"⚠️ ensure_admin_tables error: {e}")

def fetch_settings_from_db():
    """Return admin settings as dict (key->value)."""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT key, value FROM admin_settings", conn)
            if df.empty:
                return {}
            return {k: v for k, v in zip(df['key'], df['value'])}
    except Exception as e:
        print(f"⚠️ fetch_settings_from_db error: {e}")
        return {}

def fetch_caution_words_from_db():
    """Return list[str] of caution words ordered newest-first."""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT word FROM caution_words ORDER BY created_at DESC", conn)
            if df.empty:
                return []
            return df['word'].astype(str).tolist()
    except Exception as e:
        print(f"⚠️ fetch_caution_words_from_db error: {e}")
        return []

# ---------------- DB helpers for processed table ----------------
def is_processed(msg_id):
    """Check if message has been processed"""
    if not msg_id:
        return False
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM processed WHERE message_id = %s", (msg_id,))
                return cur.fetchone() is not None
    except Exception as e:
        print(f"⚠️ DB check error: {e}")
        return False

def get_processed_message_ids(message_ids):
    """Batch check which messages are already processed - PERFORMANCE OPTIMIZATION"""
    if not message_ids:
        return set()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Use IN clause for batch lookup
                cur.execute(
                    "SELECT message_id FROM processed WHERE message_id = ANY(%s)",
                    (list(message_ids),)
                )
                return {row[0] for row in cur.fetchall()}
    except Exception as e:
        print(f"⚠️ Batch DB check error: {e}")
        return set()

def mark_processed(msg_id, mailbox, sender, receivers, cc, subject, received_dt,
                    web_link, final_label, prob_neg, sender_domain):
    """Insert or update processed message"""
    if not msg_id:
        return False
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO processed
                        (message_id, mailbox, sender, receivers, cc, subject,
                         received_dt, web_link, final_label, prob_neg, sender_domain, processed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (message_id)
                    DO UPDATE SET
                        final_label = EXCLUDED.final_label,
                        prob_neg = EXCLUDED.prob_neg,
                        processed_at = EXCLUDED.processed_at
                """, (
                    msg_id,
                    mailbox or "",
                    sender or "",
                    receivers or "",
                    cc or "",
                    (subject or "")[:500],  # Truncate subject
                    received_dt or None,
                    web_link or "",
                    final_label or "Neutral",
                    float(prob_neg or 0.0),
                    sender_domain or "",
                    datetime.utcnow()
                ))
                conn.commit()
        return True
    except Exception as e:
        print(f"❌ DB insert error: {e}")
        stats['errors'] += 1
        return False

# ---------------- Auth helpers (app-only) ----------------
def acquire_token_app(client_id, tenant_id, client_secret, scopes):
    """Acquire app-only token for Microsoft Graph"""
    try:
        app = ConfidentialClientApplication(
            client_id,
            authority=f"https://login.microsoftonline.com/{tenant_id}",
            client_credential=client_secret
        )
        result = app.acquire_token_for_client(scopes=scopes)
        return result
    except Exception as e:
        print(f"❌ Token acquisition error: {e}")
        return {}

def headers_from_token(token):
    """Create headers with bearer token"""
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# ---------------- Graph helpers ----------------
def fetch_emails_for_user(access_token, user, top=50, fetch_read=False, lookback_days=None):
    """Fetch emails from user's inbox with PAGINATION support - FIXED"""
    if lookback_days is None:
        lb = LOOKBACK_DAYS
    else:
        lb = int(lookback_days)
    
    url = f"{GRAPH_V1}/users/{user}/mailFolders/Inbox/messages"
    
    # Build filter
    filters = []
    lookback_date = (datetime.utcnow() - timedelta(days=lb)).strftime('%Y-%m-%dT%H:%M:%SZ')
    filters.append(f"receivedDateTime ge {lookback_date}")
    
    if not fetch_read:
        filters.append("isRead eq false")
    
    filter_string = " and ".join(filters)
    params = {
        "$top": "999",  # Max per page (Graph API limit)
        "$filter": filter_string,
        "$select": "id,subject,from,receivedDateTime,bodyPreview,isRead",
        "$orderby": "receivedDateTime desc"
    }
    
    all_emails = []
    page_count = 0
    max_pages = (top // 999) + 1  # Calculate max pages needed
    
    try:
        while url and page_count < max_pages:
            r = requests.get(
                url, 
                headers=headers_from_token(access_token), 
                params=params if page_count == 0 else None,
                timeout=60  # Increased timeout for large responses
            )
            r.raise_for_status()
            data = r.json()
            
            page_emails = data.get("value", [])
            all_emails.extend(page_emails)
            page_count += 1
            
            print(f"   📄 Page {page_count}: fetched {len(page_emails)} emails (total: {len(all_emails)})")
            
            # Check if we've reached the limit
            if len(all_emails) >= top:
                all_emails = all_emails[:top]
                break
            
            # Get next page URL
            url = data.get("@odata.nextLink")
            params = None  # Don't use params for subsequent requests
            
            if not url or len(page_emails) == 0:
                break
        
        return all_emails
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"⚠️ Mailbox not found or no access: {user}")
        else:
            print(f"❌ HTTP error fetching emails for {user}: {e}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch emails for {user}: {e}")
        return []

def get_full_message(access_token, mailbox_owner, msg_id):
    """Fetch full message details including headers"""
    url = f"{GRAPH_V1}/users/{mailbox_owner}/messages/{msg_id}"
    params = {
        "$select": "id,subject,from,toRecipients,ccRecipients,receivedDateTime,webLink,bodyPreview,internetMessageHeaders,isRead"
    }
    try:
        r = requests.get(url, headers=headers_from_token(access_token), params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Failed to fetch full message {msg_id}: {e}")
        return None

def mark_as_read(access_token, mailbox_owner, msg_id):
    """Mark email as read in the mailbox"""
    url = f"{GRAPH_V1}/users/{mailbox_owner}/messages/{msg_id}"
    try:
        r = requests.patch(
            url,
            headers=headers_from_token(access_token),
            json={"isRead": True},
            timeout=30
        )
        r.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Failed to mark message as read: {e}")
        return False

def list_users_app_mode(access_token, top=999):
    """List all users in the tenant"""
    users = []
    url = f"{GRAPH_V1}/users"
    params = {"$select": "id,userPrincipalName,mail", "$top": str(top)}
    try:
        while url:
            r = requests.get(
                url,
                headers=headers_from_token(access_token),
                params=params if not users else None,
                timeout=30
            )
            r.raise_for_status()
            data = r.json()
            for u in data.get("value", []):
                upn = u.get("userPrincipalName") or u.get("mail")
                if upn:
                    users.append(upn)
            url = data.get("@odata.nextLink")
            params = None
        return users
    except Exception as e:
        print(f"❌ Failed to list users: {e}")
        return []

# ---------------- Utilities ----------------
def extract_domain(email_addr):
    """Extract domain from email address"""
    if not email_addr or "@" not in str(email_addr):
        return ""
    return str(email_addr).split("@")[-1].lower()

def recipients_to_csv(recipients):
    """Convert recipients list to CSV string"""
    out = []
    if not recipients:
        return ""
    # Handle different recipient formats
    if isinstance(recipients, dict) and "value" in recipients:
        recipients = recipients.get("value", [])
    if isinstance(recipients, str):
        return ",".join([a.strip() for a in recipients.replace(";", ",").split(",") if a.strip()])
    try:
        for r in recipients:
            if not r:
                continue
            if isinstance(r, str):
                addr = r.strip()
                if addr:
                    out.append(addr)
                continue
            if isinstance(r, dict):
                addr = None
                # Try different dict structures
                if "emailAddress" in r and isinstance(r["emailAddress"], dict):
                    addr = r["emailAddress"].get("address")
                elif "address" in r:
                    addr = r.get("address")
                elif "email" in r:
                    addr = r.get("email")
                if addr:
                    out.append(addr)
                    continue
                # Handle nested structures
                if "value" in r and isinstance(r["value"], list):
                    for rr in r["value"]:
                        if isinstance(rr, dict):
                            a = rr.get("emailAddress", {}).get("address") or rr.get("address")
                            if a:
                                out.append(a)
            else:
                s = str(r).strip()
                if s:
                    out.append(s)
    except Exception as e:
        print(f"⚠️ Error parsing recipients: {e}")
    # Deduplicate
    seen = set()
    deduped = []
    for a in out:
        if a and a not in seen:
            deduped.append(a)
            seen.add(a)
    return ",".join(deduped)

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')

def extract_cc(full_message, body_preview=""):
    """Extract CC recipients from message"""
    if not full_message:
        return ""
    # Try ccRecipients field first
    cc_recipients = full_message.get("ccRecipients")
    if cc_recipients:
        cc_csv = recipients_to_csv(cc_recipients)
        if cc_csv:
            return cc_csv
    # Try internet message headers
    headers = full_message.get("internetMessageHeaders") or []
    if headers and isinstance(headers, list):
        for h in headers:
            try:
                if (h.get("name") or "").lower() == "cc":
                    val = h.get("value") or ""
                    parts = [p.strip() for p in val.replace(";", ",").split(",") if p.strip()]
                    addrs = []
                    for p in parts:
                        if "<" in p and ">" in p:
                            inside = p[p.find("<")+1:p.rfind(">")].strip()
                            if "@" in inside:
                                addrs.append(inside)
                            continue
                        if "@" in p:
                            tokens = p.split()
                            for t in tokens:
                                if "@" in t:
                                    cleaned = t.strip().strip(",;")
                                    addrs.append(cleaned)
                                    break
                    if addrs:
                        seen = set()
                        deduped = []
                        for a in addrs:
                            if a not in seen:
                                deduped.append(a)
                                seen.add(a)
                        return ",".join(deduped)
                    if val.strip():
                        return val.strip()
            except Exception:
                continue
    # Last resort: parse body preview
    if not body_preview:
        body_preview = full_message.get("bodyPreview") or ""
    if body_preview and isinstance(body_preview, str):
        for line in body_preview.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if line_stripped.lower().startswith("cc:") or line_stripped.lower().startswith("c.c:"):
                after = line_stripped.split(":", 1)[1]
                emails = EMAIL_RE.findall(after)
                if emails:
                    seen = set()
                    dedup = []
                    for e in emails:
                        if e not in seen:
                            dedup.append(e)
                            seen.add(e)
                    return ",".join(dedup)
    return ""

# ---------------- Helper to safely parse settings ----------------
def safe_int(value, default):
    """Safely parse int from value"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def safe_float(value, default):
    """Safely parse float from value"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def safe_bool(value, default):
    """Safely parse bool from value"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('1', 'true', 'yes')

# ---------------- Main Processing ----------------
def process_message(access_token, mailbox, msg_basic, settings, caution_words, skip_duplicate_check=False):
    """Process a single email message"""
    mid = msg_basic.get("id")
    subj = (msg_basic.get("subject") or "")[:400]
    
    if not mid:
        return False
    
    # Skip duplicate check if already done in batch
    if not skip_duplicate_check and is_processed(mid):
        return False
    
    # Fetch full message details
    try:
        full = get_full_message(access_token, mailbox, mid)
        if not full:
            full = msg_basic
    except Exception as e:
        print(f"⚠️ Failed to fetch full metadata for {mid}: {e}")
        full = msg_basic
    
    # Extract sender
    sender = ""
    try:
        fr = full.get("from") or msg_basic.get("from")
        if isinstance(fr, dict):
            sender = fr.get("emailAddress", {}).get("address") or ""
        elif isinstance(fr, str):
            sender = fr
    except Exception:
        sender = ""
    
    # Extract recipients
    to_recipients = full.get("toRecipients") or []
    receivers_s = recipients_to_csv(to_recipients)
    
    # Extract CC
    cc_s = extract_cc(full, body_preview=(full.get("bodyPreview") or msg_basic.get("bodyPreview", "")))
    if DEBUG_CC:
        print(f"  🔍 DEBUG CC - Subject: {subj[:50]}")
        print(f"  🔍 DEBUG CC - Raw ccRecipients: {full.get('ccRecipients')}")
        print(f"  🔍 DEBUG CC - Extracted CC: '{cc_s}'")
    
    # Extract metadata
    received_dt = full.get("receivedDateTime") or msg_basic.get("receivedDateTime") or ""
    web_link = full.get("webLink") or ""
    is_read = full.get("isRead", msg_basic.get("isRead", False))
    
    # Prepare text for classification
    text_for_classify = (
        (full.get("subject", "") or subj) +
        "\n\n" +
        (full.get("bodyPreview") or msg_basic.get("bodyPreview", "") or "")
    )
    
    # Determine neg threshold from settings
    neg_thresh = safe_float(settings.get("neg_prob_thresh") if settings else None, NEG_PROB_THRESH)
    
    # Classify sentiment
    try:
        res = classify_email(
            text_for_classify,
            model_dir=MODEL_DIR,
            apply_rule=True,
            neg_prob_thresh=neg_thresh,
            caution_keywords=caution_words or []
        )
    except Exception as e:
        print(f"⚠️ Classification error for {mid}: {e}")
        res = {
            "final_label": "Neutral",
            "pred_label": "Neutral",
            "probs": [0.0, 1.0, 0.0],
            "postprocess_reason": "error"
        }
    
    final = res.get("final_label") or str(res.get("final_idx", "Neutral"))
    probs = res.get("probs", [0.0, 0.0, 0.0])
    p_neg = float(probs[0]) if len(probs) > 0 else 0.0
    sender_domain = extract_domain(sender)
    
    # Log processing
    sentiment_emoji = {"Negative": "🔴", "Neutral": "🟡", "Positive": "🟢"}.get(final, "⚪")
    print(f"  {sentiment_emoji} {subj[:80]} → {final} (neg={p_neg:.3f})")
    if cc_s:
        print(f"   📎 CC: {cc_s[:100]}")
    
    # Save to database
    success = mark_processed(
        mid, mailbox, sender, receivers_s, cc_s, subj,
        received_dt, web_link, final, p_neg, sender_domain
    )
    
    if success:
        update_stats(mailbox, final)
    
    # Optionally mark as read
    mark_read_setting = safe_bool(settings.get('mark_as_read') if settings else None, MARK_AS_READ_ENV)
    if not is_read and mark_read_setting:
        mark_as_read(access_token, mailbox, mid)
    
    return success

# ---------------- Main Loop ----------------
def main():
    print("\n" + "="*60)
    print("🚀 Email Monitor Starting - High Volume Edition")
    print("="*60)
    
    # Initialize connection pool
    print("📦 Initializing database connection pool...")
    try:
        init_connection_pool(min_conn=2, max_conn=10)
        print("✅ Connection pool initialized")
    except Exception as e:
        print(f"⚠️ Connection pool init warning: {e}")
    
    # Ensure schema exists
    print("🗄️ Ensuring database schema...")
    try:
        ensure_schema()
        print("✅ Database schema ensured")
    except Exception as e:
        print(f"❌ Schema error: {e}")
        return
    
    # Ensure admin tables exist
    ensure_admin_tables()
    
    # Acquire initial token
    print("🔐 Acquiring Microsoft Graph access token...")
    token_resp = acquire_token_app(
        CLIENT_ID, TENANT_ID, CLIENT_SECRET,
        scopes=["https://graph.microsoft.com/.default"]
    )
    if "access_token" not in token_resp:
        print(f"❌ Failed to acquire app token: {token_resp.get('error_description', token_resp)}")
        return
    
    access_token = token_resp["access_token"]
    print("✅ Access token acquired successfully")
    
    # Determine list of users to poll
    users = []
    if MAILBOX_LIST:
        users = [m.strip() for m in MAILBOX_LIST.split(",") if m.strip()]
        print(f"📬 Using mailboxes from MAILBOX_LIST: {len(users)} mailboxes")
    
    if not users:
        print("⚠️ No MAILBOX_LIST set. Attempting to list all users...")
        try:
            users = list_users_app_mode(access_token, top=999)
            print(f"✅ Found {len(users)} users in tenant")
        except Exception as e:
            print(f"❌ Failed to list users: {e}")
            return
    
    if not users:
        print("❌ No users to poll. Exiting.")
        return
    
    # Print initial config
    print("\n" + "="*60)
    print("📊 CONFIGURATION (env defaults; DB can override)")
    print("="*60)
    print(f"📬 Mailboxes: {len(users)}")
    for idx, user in enumerate(users[:10], 1):
        print(f"  {idx}. {user}")
    if len(users) > 10:
        print(f"  ... and {len(users) - 10} more")
    print(f"⏱️ Poll Interval: {POLL_INTERVAL}s")
    print(f"📧 Max Emails/Poll: {MAX_EMAILS_PER_POLL}")
    print(f"📅 Lookback Days: {LOOKBACK_DAYS}")
    print(f"👁️ Fetch Read: {FETCH_READ_EMAILS}")
    print(f"🔍 Debug CC: {DEBUG_CC}")
    print(f"🤖 Model Dir: {MODEL_DIR}")
    print(f"🔴 Neg Threshold: {NEG_PROB_THRESH}")
    print("="*60 + "\n")
    
    loop_count = 0
    
    try:
        while True:
            loop_count += 1
            loop_start = time.time()
            
            print(f"\n{'='*60}")
            print(f"🔄 CYCLE #{loop_count} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"{'='*60}")
            
            # Refresh token
            token_resp = acquire_token_app(
                CLIENT_ID, TENANT_ID, CLIENT_SECRET,
                scopes=["https://graph.microsoft.com/.default"]
            )
            if "access_token" in token_resp:
                access_token = token_resp["access_token"]
            
            # Fetch dynamic settings & caution words
            settings = fetch_settings_from_db()
            caution_words = fetch_caution_words_from_db()
            
            # Apply dynamic overrides with safe parsing
            poll_interval = safe_int(settings.get('poll_interval'), POLL_INTERVAL)
            max_emails = safe_int(settings.get('max_emails_per_poll'), MAX_EMAILS_PER_POLL)
            lookback_days = safe_int(settings.get('lookback_days'), LOOKBACK_DAYS)
            fetch_read = safe_bool(settings.get('fetch_read_emails'), FETCH_READ_EMAILS)
            neg_prob_db = safe_float(settings.get('neg_prob_thresh'), NEG_PROB_THRESH)
            mark_read_setting = safe_bool(settings.get('mark_as_read'), MARK_AS_READ_ENV)
            
            print(f"🔧 Config: poll={poll_interval}s, max={max_emails}, lookback={lookback_days}d, fetch_read={fetch_read}, mark_read={mark_read_setting}, neg_thresh={neg_prob_db}, caution_words={len(caution_words)}")
            
            cycle_processed = 0
            
            # Process each mailbox with BATCH OPTIMIZATION
            for mailbox_idx, mailbox in enumerate(users, 1):
                print(f"\n📬 [{mailbox_idx}/{len(users)}] {mailbox}")
                
                try:
                    msgs = fetch_emails_for_user(
                        access_token,
                        mailbox,
                        top=max_emails,
                        fetch_read=fetch_read,
                        lookback_days=lookback_days
                    )
                except Exception as e:
                    print(f"❌ Fetch error: {e}")
                    stats['errors'] += 1
                    time.sleep(1)
                    continue
                
                if not msgs:
                    print(f"  ℹ️ No new emails")
                    continue
                
                print(f"  📨 Found {len(msgs)} emails")
                
                # BATCH CHECK: Get all processed IDs at once
                msg_ids = [msg.get("id") for msg in msgs if msg.get("id")]
                processed_ids = get_processed_message_ids(msg_ids)
                
                # Process each message
                mailbox_processed = 0
                for msg in msgs:
                    msg_id = msg.get("id")
                    if not msg_id or msg_id in processed_ids:
                        continue
                    
                    try:
                        if process_message(access_token, mailbox, msg, settings, caution_words, skip_duplicate_check=True):
                            mailbox_processed += 1
                            cycle_processed += 1
                    except Exception as e:
                        print(f"  ❌ Error processing message: {e}")
                        stats['errors'] += 1
                
                if mailbox_processed > 0:
                    print(f"  ✅ Processed {mailbox_processed} new emails")
            
            # Cycle summary
            loop_duration = time.time() - loop_start
            print(f"\n{'='*60}")
            print(f"✅ Cycle #{loop_count} Complete")
            print(f"  📧 Processed: {cycle_processed} new emails")
            print(f"  ⏱️ Duration: {loop_duration:.2f}s")
            print(f"{'='*60}")
            
            # Print stats every 10 cycles
            if loop_count % 10 == 0:
                print_stats()
            
            # Sleep until next poll
            print(f"\n😴 Sleeping {poll_interval}s...\n")
            time.sleep(max(1, poll_interval))
            
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("⛔ SHUTDOWN - Interrupted by user")
        print("="*60)
        print_stats()
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        print_stats()

if __name__ == "__main__":
    main()
