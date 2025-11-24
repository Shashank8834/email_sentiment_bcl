# admin.py - Updated admin console with fixed caution words functions and safer callbacks
# Admin/Backend dashboard for system configuration and monitoring - PostgreSQL version

import os
from datetime import datetime, timedelta
import io
import re

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from dotenv import load_dotenv

# Import PostgreSQL config (your existing db_config)
from db_config import get_db_connection, init_connection_pool

load_dotenv()

# Configuration
PORT = int(os.environ.get("PORT", 8051))  # admin dashboard default port

# Initialize database connection pool (no-op if implemented)
try:
    init_connection_pool(min_conn=2, max_conn=10)
except Exception:
    pass

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)

app.title = "Email Monitor - Admin Console"
server = app.server  # For compatibility

# -----------------------
# New DB helpers for admin_settings & caution_words
# -----------------------
def ensure_admin_tables():
    """Create admin_settings and caution_words tables if they don't exist and seed defaults."""
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

    INSERT INTO admin_settings (key, value)
      VALUES ('neg_prob_thresh', '0.06')
    ON CONFLICT DO NOTHING;

    INSERT INTO admin_settings (key, value)
      VALUES ('poll_interval', '30'),
             ('max_emails_per_poll', '50'),
             ('auto_refresh_interval', '30')
    ON CONFLICT DO NOTHING;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SQL)
            conn.commit()
    except Exception as e:
        print("ensure_admin_tables error:", e)


def fetch_settings_from_db():
    """Return dict of key->value from admin_settings."""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT key, value FROM admin_settings", conn)
            return {k: v for k, v in zip(df['key'], df['value'])}
    except Exception as e:
        print("fetch_settings_from_db error:", e)
        return {}


def set_setting(key: str, value: str) -> bool:
    """Upsert a setting into admin_settings."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO admin_settings (key, value, updated_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
                """, (key, str(value)))
                conn.commit()
                return True
    except Exception as e:
        print("set_setting error:", e)
        return False


# --- Caution words helpers (fixed, improved) ---

def add_caution_word(word: str):
    """
    Insert a caution word (normalized to lower). Returns (ok, msg).
    FIXED: Better validation and error messages
    """
    if not word or not str(word).strip():
        return False, "empty"
    
    # Normalize to lowercase and trim
    w = str(word).strip().lower()
    
    # Validate word is not just spaces or special chars
    if len(w) < 2:
        return False, "Word must be at least 2 characters"
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check if exists first
                cur.execute("SELECT id FROM caution_words WHERE word = %s", (w,))
                if cur.fetchone():
                    return False, "exists"
                
                # Insert new word
                cur.execute(
                    "INSERT INTO caution_words (word, created_at) VALUES (%s, now()) RETURNING id", 
                    (w,)
                )
                word_id = cur.fetchone()
                conn.commit()
                
                if word_id:
                    print(f"✅ Added caution word: '{w}' (id={word_id[0]})")
                    return True, "added"
                else:
                    return False, "insert_failed"
                    
    except Exception as e:
        print(f"❌ add_caution_word error: {e}")
        return False, str(e)


def get_caution_words():
    """
    Return list of dicts: {id, word, created_at} ordered newest first.
    FIXED: Better error handling and logging
    """
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query(
                "SELECT id, word, created_at FROM caution_words ORDER BY created_at DESC", 
                conn
            )
            
            if df.empty:
                print("⚠️ No caution words in database")
                return []
            
            print(f"✅ Retrieved {len(df)} caution words from database")
            return df.to_dict("records")
            
    except Exception as e:
        print(f"❌ get_caution_words error: {e}")
        return []


def fetch_caution_words_from_db():
    """
    Return list[str] of caution words ordered newest-first.
    FIXED: Ensures lowercase normalization
    """
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query(
                "SELECT word FROM caution_words ORDER BY created_at DESC", 
                conn
            )
            
            if df.empty:
                print("⚠️ No caution words found, using defaults")
                return []
            
            # Ensure lowercase and clean
            words = [str(w).strip().lower() for w in df['word'].tolist() if w and str(w).strip()]
            print(f"✅ Loaded {len(words)} caution keywords: {words[:5]}...")
            return words
            
    except Exception as e:
        print(f"❌ fetch_caution_words_from_db error: {e}")
        return []


def remove_caution_word(word_id: int) -> bool:
    """Delete caution word by id."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM caution_words WHERE id = %s", (int(word_id),))
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        print("remove_caution_word error:", e)
        return False


# -----------------------
# Existing DB / app code
# -----------------------

def load_db(max_rows=5000):
    """Load data from PostgreSQL database"""
    try:
        with get_db_connection() as conn:
            query = """
                SELECT message_id, mailbox, sender, receivers, cc, subject,
                       final_label, prob_neg, web_link, sender_domain,
                       processed_at, received_dt
                FROM processed
                ORDER BY processed_at DESC
                LIMIT %s
            """
            df = pd.read_sql_query(query, conn, params=(max_rows,))
            print(f"✅ load_db: Retrieved {len(df)} rows from database")
    except Exception as e:
        print(f"❌ Failed to read from PostgreSQL: {e}")
        return pd.DataFrame(columns=[
            "message_id", "mailbox", "sender", "receivers", "cc", "subject",
            "final_label", "prob_neg", "web_link", "sender_domain",
            "processed_at", "received_dt"
        ])
    
    if not df.empty:
        # Ensure datetime columns are parsed
        if 'processed_at' in df.columns:
            df['processed_at'] = pd.to_datetime(df['processed_at'], errors='coerce')
        if 'received_dt' in df.columns:
            df['received_dt'] = pd.to_datetime(df['received_dt'], errors='coerce')
        
        # Extract client domain
        df["client_domain"] = df.get("sender_domain",
            df["sender"].apply(lambda s: (s or "").split("@")[-1].lower() if "@" in (s or "") else "")
        ).fillna("unknown").astype(str)
    
    return df


def get_db_stats():
    """Get database statistics from PostgreSQL"""
    stats = {
        'total_records': 0,
        'first_processed': None,
        'last_processed': None,
        'unique_mailboxes': 0,
        'unique_senders': 0,
        'unique_domains': 0
    }
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM processed")
                stats['total_records'] = cur.fetchone()[0]
                
                cur.execute("SELECT MIN(processed_at), MAX(processed_at) FROM processed")
                first, last = cur.fetchone()
                stats['first_processed'] = first
                stats['last_processed'] = last
                
                cur.execute("SELECT COUNT(DISTINCT mailbox) FROM processed")
                stats['unique_mailboxes'] = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(DISTINCT sender) FROM processed")
                stats['unique_senders'] = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(DISTINCT sender_domain) FROM processed")
                stats['unique_domains'] = cur.fetchone()[0]
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
    
    return stats

# Configuration
IST_OFFSET = timedelta(hours=5, minutes=30)

def to_ist(dt_series):
    """Convert UTC datetime series to IST for display"""
    if dt_series is None:
        return dt_series
    return pd.to_datetime(dt_series) + IST_OFFSET

def format_ist(dt_series, format_str='%Y-%m-%d %H:%M IST'):
    """Format datetime series as IST string"""
    ist_times = to_ist(dt_series)
    return ist_times.dt.strftime(format_str)

def get_current_ist():
    """Get current time in IST"""
    return datetime.utcnow() + IST_OFFSET

# Navbar
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.I(className="fas fa-cog me-2"),
                dbc.NavbarBrand("Email Monitor - Admin Console", className="ms-2")
            ])
        ], align="center", className="g-0"),
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [html.I(className="fas fa-sync-alt me-2"), "Refresh"],
                    id="admin-refresh-button",
                    color="light",
                    size="sm",
                    outline=True
                )
            ], align="center")
        ])
    ], fluid=True),
    color="dark",
    dark=True,
    className="mb-4"
)

# Settings panel (now reads initial values from DB)
_initial_settings = fetch_settings_from_db()
_max_rows_init = 5000
_neg_init = 0.06
if _initial_settings:
    try:
        _max_rows_init = int(_initial_settings.get('max_rows', _initial_settings.get('max_emails_per_poll', 5000)))
    except Exception:
        _max_rows_init = 5000
    try:
        _neg_init = float(_initial_settings.get('neg_prob_thresh', 0.06))
    except Exception:
        _neg_init = 0.06

settings_panel = dbc.Card([
    dbc.CardHeader("⚙️ System Settings"),
    dbc.CardBody([
        html.Label("Max Rows to Load", className="fw-bold"),
        dcc.Input(
            id='max-rows-input',
            type='number',
            value=_max_rows_init,
            min=100,
            step=500,
            className="form-control mb-3"
        ),
        html.Label("Negative Probability Threshold", className="fw-bold"),
        dcc.Slider(
            id='neg-threshold-slider',
            min=0,
            max=1,
            step=0.01,
            value=_neg_init,
            marks={0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Hr(),
        html.Label("Display Options", className="fw-bold mb-2"),
        dbc.Checklist(
            id='display-options',
            options=[
                {'label': ' Show Raw Data', 'value': 'raw'},
                {'label': ' Show System Info', 'value': 'system'},
                {'label': ' Show Debug Logs', 'value': 'debug'}
            ],
            value=['system'],
            switch=True
        ),
        html.Br(),
        dbc.Button("Save Settings", id="admin-save-settings", color="primary", className="mt-2"),
        html.Div(id='settings-save-feedback', className="mt-2")  # NEW: feedback div
    ])
], className="mb-4")

# Layout
app.layout = html.Div([
    navbar,
    dcc.Store(id='admin-data-store'),
    dcc.Store(id='settings-store', data={'max_rows': _max_rows_init, 'neg_threshold': _neg_init}),
    dcc.Interval(
        id='admin-interval',
        interval=30*1000,
        n_intervals=0,
        disabled=True
    ),
    dbc.Container([
        dbc.Row([
            dbc.Col([settings_panel], width=3),
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="📊 System Overview", tab_id="overview"),
                    dbc.Tab(label="🗄️ Database Management", tab_id="database"),
                    dbc.Tab(label="⚡ Performance Metrics", tab_id="performance"),
                    dbc.Tab(label="🔧 Configuration", tab_id="config"),
                    dbc.Tab(label="🛡️ Cautionary Words", tab_id="caution"),
                ], id="admin-tabs", active_tab="overview"),
                html.Div(id='admin-tab-content', className="mt-4")
            ], width=9)
        ])
    ], fluid=True),
    html.Hr(),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Small(id='last-updated-time', className="text-muted")
            ], width=6),
            dbc.Col([
                html.Small(id='loaded-records-info', className="text-muted")
            ], width=6)
        ])
    ], fluid=True, className="mb-3")
])

# Callback to load data
@app.callback(
    Output('admin-data-store', 'data'),
    [Input('admin-refresh-button', 'n_clicks'),
     Input('max-rows-input', 'value'),
     Input('admin-interval', 'n_intervals')]
)
def load_admin_data(n_clicks, max_rows, n_intervals):
    """Load data from database"""
    try:
        df = load_db(max_rows or 5000)
        if df.empty:
            print("⚠️ load_admin_data: DataFrame is empty")
        else:
            print(f"✅ load_admin_data: Loaded {len(df)} rows")
        return df.to_json(date_format='iso', orient='split')
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        # Return valid empty DataFrame JSON instead of None
        empty_df = pd.DataFrame(columns=[
            "message_id", "mailbox", "sender", "receivers", "cc", "subject",
            "final_label", "prob_neg", "web_link", "sender_domain",
            "processed_at", "received_dt"
        ])
        return empty_df.to_json(date_format='iso', orient='split')

# Update settings store (UI local)
@app.callback(
    Output('settings-store', 'data'),
    [Input('max-rows-input', 'value'),
     Input('neg-threshold-slider', 'value')]
)
def update_settings(max_rows, neg_threshold):
    return {'max_rows': max_rows, 'neg_threshold': neg_threshold}

# Save settings to DB - FIXED: uses different output
@app.callback(
    Output('settings-save-feedback', 'children'),
    Input('admin-save-settings', 'n_clicks'),
    State('max-rows-input', 'value'),
    State('neg-threshold-slider', 'value'),
    prevent_initial_call=True
)
def save_settings_to_db(n_clicks, max_rows, neg_threshold):
    saved1 = set_setting('max_rows', str(max_rows))
    saved2 = set_setting('neg_prob_thresh', str(neg_threshold))
    saved3 = set_setting('max_emails_per_poll', str(max_rows))
    if saved1 and saved2:
        return dbc.Alert(f"✅ Settings saved at {get_current_ist().strftime('%H:%M:%S IST')}", color="success", duration=4000)
    else:
        return dbc.Alert("❌ Failed to save settings", color="danger")

# Update footer - FIXED: single callback for footer
@app.callback(
    [Output('last-updated-time', 'children'),
     Output('loaded-records-info', 'children')],
    [Input('admin-data-store', 'data')]
)
def update_footer(data_json):
    if data_json:
        try:
            df = pd.read_json(data_json, orient='split')
            return (
                f"🕐 Last updated: {get_current_ist().strftime('%Y-%m-%d %H:%M:%S IST')}",
                f"📊 Loaded records: {len(df):,}"
            )
        except Exception as e:
            print(f"update_footer error: {e}")
    return "🕐 No data", "📊 No records"

# Render tab content - FIXED: better error handling
@app.callback(
    Output('admin-tab-content', 'children'),
    [Input('admin-tabs', 'active_tab'),
     Input('admin-data-store', 'data'),
     Input('display-options', 'value')]
)
def render_tab_content(active_tab, data_json, display_options):
    print(f"🔍 render_tab_content: tab={active_tab}, data_json type={type(data_json)}")
    
    # Handle display_options being None
    if display_options is None:
        display_options = []
    
    # Guard against None / empty store content
    if data_json is None:
        print("⚠️ data_json is None")
        return dbc.Alert("Loading data... Please wait or click Refresh.", color="info")
    
    if not data_json or str(data_json).strip() in ('', 'null', 'None'):
        print("⚠️ data_json is empty/null string")
        return dbc.Alert("No data in store. Click Refresh to load.", color="warning")
    
    try:
        df = pd.read_json(data_json, orient='split')
        print(f"✅ Parsed DataFrame: {len(df)} rows")
    except Exception as e:
        print(f"❌ Failed to parse data_json: {e}")
        return dbc.Alert(f"Error parsing data: {str(e)}. Please refresh.", color="danger")

    if df.empty:
        # Still allow access to config and caution tabs even with no data
        if active_tab == 'config':
            return create_config_tab(display_options)
        elif active_tab == 'caution':
            return create_caution_tab_ui()
        return dbc.Alert("No data available. Start the monitor to collect emails, or check database connection.", color="info")
    
    # Render appropriate tab
    if active_tab == 'overview':
        return create_overview_tab(df)
    elif active_tab == 'database':
        return create_database_tab(df, display_options)
    elif active_tab == 'performance':
        return create_performance_tab(df)
    elif active_tab == 'config':
        return create_config_tab(display_options)
    elif active_tab == 'caution':
        return create_caution_tab_ui()
    
    return html.Div("Select a tab")

# --------------------------
# Content creators
def create_overview_tab(df):
    """Create System Overview tab"""
    stats = get_db_stats()
    now_ist = get_current_ist()
    one_hour_ago = now_ist - timedelta(hours=1)
    
    df['processed_at'] = pd.to_datetime(df['processed_at'])
    recent = df[df['processed_at'] > one_hour_ago]
    processing_rate = len(recent)
    negative_count = (df["final_label"] == "Negative").sum()
    avg_neg_prob = df["prob_neg"].replace("", 0).astype(float).mean() if len(df) > 0 else 0.0
    
    first_processed = stats.get('first_processed')
    last_processed = stats.get('last_processed')
    
    first_str = format_ist(pd.Series([first_processed])).iloc[0] if first_processed else "N/A"
    last_str = format_ist(pd.Series([last_processed])).iloc[0] if last_processed else "N/A"
    
    metrics = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📧 Total Processed", className="text-muted"),
                    html.H3(f"{stats.get('total_records', 0):,}")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📬 Unique Mailboxes", className="text-muted"),
                    html.H3(f"{stats.get('unique_mailboxes', 0)}")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("👤 Unique Senders", className="text-muted"),
                    html.H3(f"{stats.get('unique_senders', 0):,}")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🏢 Unique Domains", className="text-muted"),
                    html.H3(f"{stats.get('unique_domains', 0)}")
                ])
            ])
        ], width=3),
    ], className="mb-4")
    
    metrics2 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("⚡ Emails/Hour", className="text-muted"),
                    html.H3(f"{processing_rate}")
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🔴 Negative Count", className="text-muted"),
                    html.H3(f"{negative_count:,}")
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📈 Avg Neg Probability", className="text-muted"),
                    html.H3(f"{avg_neg_prob:.4f}")
                ])
            ])
        ], width=4),
    ], className="mb-4")
    
    # Processing activity chart
    df['hour'] = df['processed_at'].dt.floor('H')
    hourly_counts = df.groupby('hour').size().reset_index(name='count')
    hourly_counts = hourly_counts.sort_values('hour').tail(24)
    
    fig_activity = px.line(
        hourly_counts,
        x='hour',
        y='count',
        title="Recent Processing Activity (Last 24 Hours)",
        labels={'hour': 'Time (IST)', 'count': 'Number of Emails'}
    )
    fig_activity.update_layout(height=300)
    
    # Sentiment distribution
    sentiment_counts = df['final_label'].value_counts()
    fig_sentiment = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        title="Overall Sentiment Distribution",
        labels={'x': 'Sentiment', 'y': 'Count'},
        color=sentiment_counts.index,
        color_discrete_map={'Negative': '#ff4444', 'Neutral': '#ffa500', 'Positive': '#00cc00'}
    )
    fig_sentiment.update_layout(height=300, showlegend=False)
    
    # Additional info cards
    info_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("⏱️ First Processed", className="text-muted"),
                    html.H5(first_str)
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🔄 Last Processed", className="text-muted"),
                    html.H5(last_str)
                ])
            ])
        ], width=6),
    ], className="mb-4")
    
    return html.Div([
        html.H4("System Overview", className="mb-3"),
        metrics,
        info_cards,
        metrics2,
        html.Hr(),
        dcc.Graph(figure=fig_activity),
        html.Hr(),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_sentiment)], width=6),
            dbc.Col([
                html.H5("Sentiment Distribution Table"),
                dash_table.DataTable(
                    data=pd.DataFrame({
                        'Sentiment': sentiment_counts.index,
                        'Count': sentiment_counts.values,
                        'Percentage': (sentiment_counts.values / sentiment_counts.sum() * 100).round(2)
                    }).to_dict('records'),
                    columns=[
                        {'name': 'Sentiment', 'id': 'Sentiment'},
                        {'name': 'Count', 'id': 'Count'},
                        {'name': 'Percentage', 'id': 'Percentage'}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                )
            ], width=6)
        ])
    ])


def create_database_tab(df, display_options):
    """Create Database Management tab"""
    stats = get_db_stats()
    
    # Data quality checks
    missing_subjects = df['subject'].isna().sum()
    missing_senders = df['sender'].isna().sum()
    missing_labels = df['final_label'].isna().sum()
    
    quality_alerts = []
    if missing_subjects > 0:
        quality_alerts.append(dbc.Alert(f"⚠️ {missing_subjects} emails with missing subjects", color="warning"))
    if missing_senders > 0:
        quality_alerts.append(dbc.Alert(f"⚠️ {missing_senders} emails with missing senders", color="warning"))
    if missing_labels > 0:
        quality_alerts.append(dbc.Alert(f"🔴 {missing_labels} emails with missing sentiment labels", color="danger"))
    if not quality_alerts:
        quality_alerts.append(dbc.Alert("✅ No data quality issues detected", color="success"))
    
    # Show raw data if requested
    content = [
        html.H4("Database Management", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Database Information"),
                    dbc.CardBody([
                        html.P([html.Strong("Database Type: "), html.Code("PostgreSQL")]),
                        html.P([html.Strong("Total Records: "), f"{stats.get('total_records', 0):,}"]),
                        html.P([html.Strong("First Record: "), str(stats.get('first_processed', 'N/A'))]),
                        html.P([html.Strong("Last Record: "), str(stats.get('last_processed', 'N/A'))]),
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Data Quality Checks"),
                    dbc.CardBody(quality_alerts)
                ])
            ], width=6)
        ], className="mb-4")
    ]
    
    if 'raw' in display_options:
        content.extend([
            html.Hr(),
            html.H5("Raw Data Sample (First 50 rows)"),
            dash_table.DataTable(
                data=df.head(50).to_dict('records'),
                columns=[{'name': col, 'id': col} for col in df.columns],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '6px'}
            )
        ])
    
    return html.Div(content)


def create_performance_tab(df):
    """Create Performance Metrics tab"""
    df['date'] = pd.to_datetime(df['processed_at']).dt.date
    daily_counts = df.groupby('date').size().reset_index(name='count')
    daily_counts = daily_counts.sort_values('date').tail(30)
    
    # Convert dates to IST for display
    daily_counts['date_ist'] = to_ist(pd.to_datetime(daily_counts['date'])).dt.strftime('%Y-%m-%d')
    
    fig_timeline = px.line(
        daily_counts,
        x='date_ist',
        y='count',
        title="Processing Timeline (Last 30 Days)",
        labels={'date_ist': 'Date', 'count': 'Number of Emails'}
    )
    fig_timeline.update_layout(height=400)
    
    # Top mailboxes
    mailbox_counts = df['mailbox'].value_counts().head(10)
    fig_mailbox = px.bar(
        x=mailbox_counts.values,
        y=mailbox_counts.index,
        orientation='h',
        title="Top 10 Mailbox Activity",
        labels={'x': 'Number of Emails', 'y': 'Mailbox'}
    )
    fig_mailbox.update_layout(height=400)
    
    # Domain performance
    domain_counts = df['client_domain'].value_counts().head(10)
    fig_domain = px.bar(
        x=domain_counts.values,
        y=domain_counts.index,
        orientation='h',
        title="Top 10 Client Domains",
        labels={'x': 'Number of Emails', 'y': 'Client Domain'}
    )
    fig_domain.update_layout(height=400)
    
    return html.Div([
        html.H4("Performance Metrics", className="mb-3"),
        dcc.Graph(figure=fig_timeline),
        html.Hr(),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_mailbox)], width=6),
            dbc.Col([dcc.Graph(figure=fig_domain)], width=6)
        ])
    ])


def create_config_tab(display_options):
    """Create Configuration tab"""
    env_vars = {
        'CLIENT_ID': os.getenv('CLIENT_ID', 'Not set'),
        'TENANT_ID': os.getenv('TENANT_ID', 'Not set'),
        'DATABASE_URL': os.getenv('DATABASE_URL', 'Not set'),
        'POLL_INTERVAL': os.getenv('POLL_INTERVAL', 'Not set'),
        'MODEL_DIR': os.getenv('INFERENCE_MODEL_DIR', 'Not set'),
        'DEBUG_CC': os.getenv('DEBUG_CC', 'Not set')
    }
    
    # Mask sensitive values
    masked_vars = {}
    for key, value in env_vars.items():
        if key in ['CLIENT_SECRET', 'DATABASE_URL']:
            masked_vars[key] = value[:20] + "..." if value != 'Not set' and len(value) > 20 else value
        else:
            masked_vars[key] = value
    
    env_content = []
    if 'system' in display_options:
        env_content = [
            html.H5("Environment Variables", className="mb-3"),
            dash_table.DataTable(
                data=pd.DataFrame(list(masked_vars.items()), columns=['Variable', 'Value']).to_dict('records'),
                columns=[
                    {'name': 'Variable', 'id': 'Variable'},
                    {'name': 'Value', 'id': 'Value'}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
        ]
    
    # Debug logs section
    debug_content = []
    if 'debug' in display_options:
        debug_content = [
            html.Hr(),
            html.H5("Debug Logs", className="mb-3"),
            dbc.Alert("Debug logging is enabled. Check console for detailed logs.", color="info"),
            dbc.Button("Test Database Connection", id="test-db-btn", color="primary", className="mb-3"),
            html.Div(id="db-test-result")
        ]
    
    return html.Div([
        html.H4("Configuration Management", className="mb-3"),
        html.Div(env_content),
        *debug_content
    ])


def create_caution_tab_ui():
    """Render caution words tab UI"""
    words = get_caution_words()
    df = pd.DataFrame(words) if words else pd.DataFrame(columns=['id', 'word', 'created_at'])
    return html.Div([
        html.H4("Cautionary Words", className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Input(id='caution-new-word', placeholder='Enter a word or phrase', type='text', className='form-control'), width=8),
            dbc.Col(dbc.Button('Add', id='caution-add-btn', color='primary', className='w-100'), width=2),
            dbc.Col(dbc.Button('Delete Selected', id='caution-delete-btn', color='danger', className='w-100'), width=2),
        ], className='mb-3'),
        dash_table.DataTable(
            id='caution-words-table',
            data=df.to_dict('records'),
            columns=[
                {'name': 'ID', 'id': 'id'},
                {'name': 'Word / Phrase', 'id': 'word'},
                {'name': 'Added', 'id': 'created_at'}
            ],
            row_selectable='multi',
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '6px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        ),
        html.Div(id='caution-feedback', className='mt-2')
    ])

# FIXED callback for caution words management
@app.callback(
    [Output('caution-words-table', 'data'),
     Output('caution-feedback', 'children'),
     Output('caution-new-word', 'value')],  # Clear input on success
    [Input('caution-add-btn', 'n_clicks'),
     Input('caution-delete-btn', 'n_clicks')],
    [State('caution-new-word', 'value'),
     State('caution-words-table', 'selected_rows'),
     State('caution-words-table', 'data')],
    prevent_initial_call=True
)
def handle_caution_actions(add_n, delete_n, new_word, selected_rows, table_data):
    """
    FIXED: Better feedback and input clearing
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        words = get_caution_words()
        return words, "", ""
    
    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trig == 'caution-add-btn':
        if not new_word or not new_word.strip():
            return table_data or [], dbc.Alert(
                "⚠️ Please enter a word or phrase", 
                color="warning", 
                duration=3000
            ), dash.no_update
        
        ok, msg = add_caution_word(new_word)
        
        if ok:
            words = get_caution_words()
            return words, dbc.Alert(
                f"✅ Added: '{new_word}' (normalized to lowercase)", 
                color="success", 
                duration=3000
            ), ""  # Clear input
        else:
            if msg == "exists":
                return table_data or [], dbc.Alert(
                    f"⚠️ Word already exists: '{new_word}'", 
                    color="warning", 
                    duration=3000
                ), dash.no_update
            elif msg == "empty":
                return table_data or [], dbc.Alert(
                    "⚠️ Please enter a non-empty word", 
                    color="warning", 
                    duration=3000
                ), dash.no_update
            else:
                return table_data or [], dbc.Alert(
                    f"❌ Failed to add: {msg}", 
                    color="danger", 
                    duration=4000
                ), dash.no_update
    
    elif trig == 'caution-delete-btn':
        if not selected_rows:
            return table_data or [], dbc.Alert(
                "⚠️ No rows selected for deletion", 
                color="warning", 
                duration=3000
            ), dash.no_update
        
        ids_to_remove = []
        words_to_remove = []
        
        for idx in selected_rows:
            try:
                row = table_data[int(idx)]
                ids_to_remove.append(row.get('id'))
                words_to_remove.append(row.get('word', 'unknown'))
            except Exception as e:
                print(f"⚠️ Error processing row {idx}: {e}")
                continue
        
        removed = 0
        for rid in ids_to_remove:
            if remove_caution_word(rid):
                removed += 1
        
        words = get_caution_words()
        
        if removed > 0:
            return words, dbc.Alert(
                f"🗑️ Removed {removed} word(s): {', '.join(words_to_remove)}", 
                color="info", 
                duration=4000
            ), dash.no_update
        else:
            return table_data or [], dbc.Alert(
                "❌ Failed to remove selected words", 
                color="danger", 
                duration=3000
            ), dash.no_update
    
    return table_data or [], "", dash.no_update

# Diagnostic function to test caution words
def test_caution_words_system():
    """
    Test function to verify caution words are working
    Call this from admin interface or startup
    """
    print("\n" + "="*60)
    print("🧪 Testing Caution Words System")
    print("="*60)
    
    # Test database retrieval
    print("\n1. Testing database retrieval...")
    words = fetch_caution_words_from_db()
    print(f"   Found {len(words)} words: {words[:10]}")
    
    # Test inference integration
    print("\n2. Testing inference integration...")
    try:
        from monitor.inference_local import classify_email
        
        test_texts = [
            "I am concerned about the delays",
            "There are gaps in the documentation",
            "Everything looks great, thank you!"
        ]
        
        for text in test_texts:
            result = classify_email(
                text, 
                apply_rule=True, 
                caution_keywords=words,
                neg_prob_thresh=0.06
            )
            print(f"\n   Text: {text[:50]}...")
            print(f"   Model: {result['pred_label']} -> Final: {result['final_label']}")
            print(f"   Reason: {result['postprocess_reason']}")
    except Exception as e:
        print(f"   ❌ Inference test failed: {e}")
    
    print("\n" + "="*60)
    print("✅ Caution words test complete")
    print("="*60 + "\n")

# Test DB connection callback (for debug mode)
@app.callback(
    Output('db-test-result', 'children'),
    Input('test-db-btn', 'n_clicks'),
    prevent_initial_call=True
)
def test_db_connection(n_clicks):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                if result:
                    return dbc.Alert("✅ Database connection successful!", color="success")
    except Exception as e:
        return dbc.Alert(f"❌ Database connection failed: {str(e)}", color="danger")
    return dbc.Alert("❌ Unknown error", color="danger")

# Initialize admin tables on startup
try:
    ensure_admin_tables()
    print("✅ Admin tables initialized")
except Exception as e:
    print(f"❌ ensure_admin_tables failed: {e}")

# Test load on startup
try:
    test_df = load_db(10)
    print(f"🧪 Startup test load: {len(test_df)} rows")
    if not test_df.empty:
        print(f"🧪 Columns: {list(test_df.columns)}")
except Exception as e:
    print(f"🧪 Startup test load FAILED: {e}")

# Run app
if __name__ == '__main__':
    @app.server.route("/_health")
    def healthcheck():
        return "OK", 200

    print(f"🚀 Starting Admin Dashboard on port {PORT}")
    app.run(debug=False, host='0.0.0.0', port=PORT)
