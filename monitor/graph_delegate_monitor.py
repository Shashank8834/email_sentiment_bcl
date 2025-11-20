# monitor/graph_delegate_monitor.py
# App-only Graph monitor that polls mailboxes and writes to PostgreSQL

import os
import time
import requests
import re
from datetime import datetime
from msal import ConfidentialClientApplication
from db_config import get_db_connection, ensure_schema, init_connection_pool

# ---------------- CONFIG from ENV ----------------
CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
DEBUG_CC = os.getenv("DEBUG_CC", "False").lower() in ("1","true","yes")
MAILBOX_LIST = os.getenv("MAILBOX_LIST", "")
MODEL_DIR = os.getenv("INFERENCE_MODEL_DIR", "/data/model")
# -------------------------------------------------

if not CLIENT_ID or not TENANT_ID or not CLIENT_SECRET:
    raise SystemExit("CLIENT_ID, TENANT_ID and CLIENT_SECRET must be set.")

GRAPH_V1 = "https://graph.microsoft.com/v1.0"

# Try to import local classifier
try:
    from inference_local import classify_email
except Exception as e:
    print("Warning: inference_local.classify_email not found; using fallback.", e)
    def classify_email(text, **kwargs):
        t = (text or "").lower()
        if any(x in t for x in ("disappoint", "unresolved", "delay", "not happy", "unable", "concern")):
            return {"final_label":"Negative","pred_label":"Negative","probs":[0.9,0.08,0.02],"postprocess_reason":"fallback"}
        if any(x in t for x in ("thank", "great", "well done", "congrat")):
            return {"final_label":"Positive","pred_label":"Positive","probs":[0.01,0.05,0.94],"postprocess_reason":"fallback"}
        return {"final_label":"Neutral","pred_label":"Neutral","probs":[0.05,0.9,0.05],"postprocess_reason":"fallback"}

# ---------------- DB helpers ----------------
def is_processed(msg_id):
    """Check if message has been processed"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM processed WHERE message_id = %s", (msg_id,))
            return cur.fetchone() is not None

def mark_processed(msg_id, mailbox, sender, receivers, cc, subject, received_dt, 
                   web_link, final_label, prob_neg, sender_domain):
    """Insert or update processed message"""
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
                msg_id, mailbox, sender or "", receivers or "", cc or "", 
                subject or "", received_dt or None, web_link or "", 
                final_label or "", float(prob_neg or 0.0), 
                sender_domain or "", datetime.utcnow()
            ))

# ---------------- Auth helpers (app-only) ----------------
def acquire_token_app(client_id, tenant_id, client_secret, scopes):
    app = ConfidentialClientApplication(
        client_id, 
        authority=f"https://login.microsoftonline.com/{tenant_id}", 
        client_credential=client_secret
    )
    result = app.acquire_token_for_client(scopes=scopes)
    return result

def headers_from_token(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# ---------------- Graph helpers ----------------
def fetch_unread_for_user(access_token, user, top=50):
    url = f"{GRAPH_V1}/users/{user}/mailFolders/Inbox/messages"
    params = {
        "$top": str(top),
        "$filter": "isRead eq false",
        "$select": "id,subject,from,receivedDateTime,bodyPreview"
    }
    r = requests.get(url, headers=headers_from_token(access_token), params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("value", [])

def get_full_message(access_token, mailbox_owner, msg_id):
    url = f"{GRAPH_V1}/users/{mailbox_owner}/messages/{msg_id}"
    params = {"$select":"id,subject,from,toRecipients,ccRecipients,receivedDateTime,webLink,bodyPreview,internetMessageHeaders"}
    r = requests.get(url, headers=headers_from_token(access_token), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def list_users_app_mode(access_token, top=999):
    users = []
    url = f"{GRAPH_V1}/users"
    params = {"$select":"id,userPrincipalName,mail", "$top": str(top)}
    while url:
        r = requests.get(url, headers=headers_from_token(access_token), 
                        params=params if not users else None, timeout=30)
        r.raise_for_status()
        data = r.json()
        for u in data.get("value", []):
            upn = u.get("userPrincipalName") or u.get("mail")
            if upn:
                users.append(upn)
        url = data.get("@odata.nextLink")
        params = None
    return users

# ---------------- Utilities ----------------
def extract_domain(email_addr):
    if not email_addr or "@" not in email_addr:
        return ""
    return email_addr.split("@")[-1].lower()

def recipients_to_csv(recipients):
    out = []
    if not recipients:
        return ""
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
                if "emailAddress" in r and isinstance(r["emailAddress"], dict):
                    addr = r["emailAddress"].get("address")
                elif "address" in r:
                    addr = r.get("address")
                elif "email" in r:
                    addr = r.get("email")
                if addr:
                    out.append(addr)
                    continue
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
    except Exception:
        pass
    seen = set()
    deduped = []
    for a in out:
        if a and a not in seen:
            deduped.append(a)
            seen.add(a)
    return ",".join(deduped)

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')

def extract_cc(full_message, body_preview=""):
    cc_recipients = None
    try:
        cc_recipients = full_message.get("ccRecipients")
    except Exception:
        cc_recipients = None
    if cc_recipients:
        cc_csv = recipients_to_csv(cc_recipients)
        if cc_csv:
            return cc_csv
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
                        seen = set(); deduped = []
                        for a in addrs:
                            if a not in seen:
                                deduped.append(a); seen.add(a)
                        return ",".join(deduped)
                    if val.strip():
                        return val.strip()
            except Exception:
                continue
    if not body_preview:
        body_preview = full_message.get("bodyPreview") or ""
    if body_preview and isinstance(body_preview, str):
        for line in body_preview.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if line_stripped.lower().startswith("cc:") or line_stripped.lower().startswith("c.c:"):
                after = line_stripped.split(":",1)[1]
                emails = EMAIL_RE.findall(after)
                if emails:
                    seen=set(); dedup=[]
                    for e in emails:
                        if e not in seen:
                            dedup.append(e); seen.add(e)
                    return ",".join(dedup)
        emails_all = EMAIL_RE.findall(body_preview)
        if emails_all:
            seen=set(); dedup=[]
            for e in emails_all:
                if e not in seen:
                    dedup.append(e); seen.add(e)
            if len(dedup) >= 2:
                return ",".join(dedup)
    return ""

# ---------------- Main ----------------
def main():
    print("🚀 Initializing Email Monitor with PostgreSQL...")
    
    # Initialize connection pool
    init_connection_pool(min_conn=2, max_conn=10)
    
    # Ensure schema exists
    ensure_schema()

    # Acquire initial token
    token_resp = acquire_token_app(CLIENT_ID, TENANT_ID, CLIENT_SECRET, 
                                   scopes=["https://graph.microsoft.com/.default"])
    if "access_token" not in token_resp:
        print("❌ Failed to acquire app token:", token_resp)
        return
    access_token = token_resp["access_token"]

    # Determine list of users to poll
    users = []
    if MAILBOX_LIST:
        users = [m.strip() for m in MAILBOX_LIST.split(",") if m.strip()]
    if not users:
        print("No MAILBOX_LIST set. Attempting to list all users...")
        try:
            users = list_users_app_mode(access_token, top=999)
        except Exception as e:
            print(f"❌ Failed to list users: {e}")
            return
    if not users:
        print("❌ No users to poll. Exiting.")
        return

    print(f"📬 Monitoring {len(users)} mailboxes: {users}")

    try:
        while True:
            # Refresh token each loop
            token_resp = acquire_token_app(CLIENT_ID, TENANT_ID, CLIENT_SECRET, 
                                          scopes=["https://graph.microsoft.com/.default"])
            access_token = token_resp.get("access_token", access_token)

            for mailbox in users:
                try:
                    msgs = fetch_unread_for_user(access_token, mailbox, top=50)
                except Exception as e:
                    print(f"❌ Failed to fetch messages for {mailbox}: {e}")
                    time.sleep(1)
                    continue

                if not msgs:
                    continue
                print(f"📨 [{mailbox}] fetched {len(msgs)} unread")

                for m in msgs:
                    mid = m.get("id")
                    subj = (m.get("subject") or "")[:400]
                    if not mid:
                        continue
                    if is_processed(mid):
                        continue

                    try:
                        full = get_full_message(access_token, mailbox, mid)
                    except Exception as e:
                        print(f"⚠️ Failed to fetch full metadata: {e}")
                        full = m

                    sender = ""
                    try:
                        fr = full.get("from") or m.get("from")
                        if isinstance(fr, dict):
                            sender = fr.get("emailAddress", {}).get("address") or ""
                        elif isinstance(fr, str):
                            sender = fr
                    except Exception:
                        sender = ""

                    to_recipients = full.get("toRecipients") or []
                    receivers_s = recipients_to_csv(to_recipients)
                    cc_s = extract_cc(full, body_preview=(full.get("bodyPreview") or m.get("bodyPreview","")))

                    if DEBUG_CC:
                        print(f"  🔍 DEBUG CC - Subject: {subj[:50]}")
                        print(f"  🔍 DEBUG CC - Raw ccRecipients: {full.get('ccRecipients')}")
                        print(f"  🔍 DEBUG CC - Extracted CC: '{cc_s}'")

                    received_dt = full.get("receivedDateTime") or m.get("receivedDateTime") or ""
                    web_link = full.get("webLink") or ""

                    text_for_classify = (full.get("subject","") or subj) + "\n\n" + (full.get("bodyPreview") or m.get("bodyPreview","") or "")
                    try:
                        res = classify_email(text_for_classify, model_dir=MODEL_DIR)
                    except Exception as e:
                        print(f"⚠️ Classification error: {e}")
                        res = {"final_label":"Neutral","pred_label":"Neutral","probs":[0.0,1.0,0.0], "postprocess_reason":"error"}

                    final = res.get("final_label") or str(res.get("final_idx"))
                    probs = res.get("probs", [0.0,0.0,0.0])
                    p_neg = float(probs[0]) if len(probs)>0 else 0.0
                    sender_domain = extract_domain(sender)

                    print(f"✅ [{mailbox}] {subj[:100]} → {final} (neg_prob={p_neg:.3f}) domain={sender_domain}")
                    if cc_s:
                        print(f"   CC: {cc_s}")

                    mark_processed(mid, mailbox, sender, receivers_s, cc_s, subj, 
                                 received_dt, web_link, final, p_neg, sender_domain)

            print(f"⏱️ Sleeping for {POLL_INTERVAL} seconds...")
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user – exiting.")

if __name__ == "__main__":
    main()