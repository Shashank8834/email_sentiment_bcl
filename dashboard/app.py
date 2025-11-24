# dashboard/app.py (patched)
# Dash dashboard with PostgreSQL support - Full Features Restored
# Fixed OTP-based login with Mailjet
# Added: dynamic admin_settings + caution_words integration (live)

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# DB helpers (must exist in your repo)
from db_config import get_db_connection, init_connection_pool

# Mailjet Configuration
try:
    from mailjet_rest import Client
    MAILJET_AVAILABLE = True
except ImportError:
    MAILJET_AVAILABLE = False
    print("⚠️ Warning: mailjet_rest not installed. Run: pip install mailjet-rest")

load_dotenv()

# Config
AUTO_REFRESH_INTERVAL_SECONDS = 30
IST_OFFSET = timedelta(hours=5, minutes=30)

# OTP Configuration
OTP_EXPIRY_MINUTES = 5
OTP_LENGTH = 6

# Initialize app
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)
server = app.server
app.title = "Email Sentiment Analytics"

# Initialize database connection pool
init_connection_pool(min_conn=2, max_conn=10)

# ---------------- Mailjet Helper Functions ----------------

def validate_mailjet_config():
    """Validate that Mailjet is properly configured"""
    if not MAILJET_AVAILABLE:
        return False, "Mailjet library not installed"
    
    api_key = os.getenv('MAILJET_API_KEY')
    api_secret = os.getenv('MAILJET_API_SECRET')
    sender_email = os.getenv('MAILJET_SENDER_EMAIL')
    
    if not api_key or api_key.strip() == '':
        return False, "MAILJET_API_KEY not set in .env file"
    
    if not api_secret or api_secret.strip() == '':
        return False, "MAILJET_API_SECRET not set in .env file"
    
    if not sender_email or sender_email.strip() == '':
        return False, "MAILJET_SENDER_EMAIL not set in .env file"
    
    if '@' not in sender_email:
        return False, "MAILJET_SENDER_EMAIL is not a valid email address"
    
    return True, "Mailjet configured successfully"

def generate_otp():
    """Generate a secure random OTP"""
    return ''.join([str(random.randint(0, 9)) for _ in range(OTP_LENGTH)])

def send_otp_via_mailjet(email, otp):
    """Send OTP email using Mailjet API"""
    if not MAILJET_AVAILABLE:
        return False, "Mailjet library not installed. Please install with: pip install mailjet-rest"
    
    try:
        api_key = os.getenv('MAILJET_API_KEY', '').strip()
        api_secret = os.getenv('MAILJET_API_SECRET', '').strip()
        sender_email = os.getenv('MAILJET_SENDER_EMAIL', '').strip()
        
        if not api_key:
            return False, "MAILJET_API_KEY not configured in .env file"
        
        if not api_secret:
            return False, "MAILJET_API_SECRET not configured in .env file"
        
        if not sender_email:
            return False, "MAILJET_SENDER_EMAIL not configured in .env file"
        
        mailjet = Client(auth=(api_key, api_secret), version='v3.1')
        
        data = {
            'Messages': [
                {
                    "From": {
                        "Email": sender_email,
                        "Name": "Email Analytics Dashboard"
                    },
                    "To": [
                        {
                            "Email": email,
                            "Name": "Dashboard User"
                        }
                    ],
                    "Subject": "Your Login OTP - Email Analytics Dashboard",
                    "TextPart": f"""Your One-Time Password (OTP) is: {otp}

This OTP is valid for {OTP_EXPIRY_MINUTES} minutes.

If you didn't request this login, please ignore this email or contact your administrator.

---
Email Analytics Dashboard
""",
                    "HTMLPart": f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="utf-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    </head>
                    <body style="margin: 0; padding: 0; font-family: Arial, sans-serif; background-color: #f4f4f4;">
                        <table role="presentation" style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <td align="center" style="padding: 40px 0;">
                                    <table role="presentation" style="width: 600px; border-collapse: collapse; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                        <tr>
                                            <td style="padding: 40px 30px;">
                                                <h2 style="color: #333333; margin: 0 0 20px 0; font-size: 24px;">🔐 Email Analytics Dashboard</h2>
                                                <p style="color: #666666; font-size: 16px; line-height: 1.5; margin: 0 0 30px 0;">
                                                    Your One-Time Password for secure login:
                                                </p>
                                                <div style="background-color: #f8f9fa; padding: 30px; border-radius: 5px; text-align: center; margin: 0 0 30px 0;">
                                                    <p style="font-size: 18px; color: #666666; margin: 0 0 15px 0;">Your OTP is:</p>
                                                    <h1 style="font-size: 42px; color: #007bff; letter-spacing: 8px; margin: 0; font-weight: bold;">{otp}</h1>
                                                    <p style="color: #999999; font-size: 14px; margin: 20px 0 0 0;">
                                                        Valid for {OTP_EXPIRY_MINUTES} minutes
                                                    </p>
                                                </div>
                                                <p style="color: #666666; font-size: 14px; line-height: 1.5; margin: 0 0 20px 0;">
                                                    Enter this code in the dashboard to complete your login.
                                                </p>
                                                <hr style="border: none; border-top: 1px solid #eeeeee; margin: 30px 0;">
                                                <p style="color: #999999; font-size: 12px; line-height: 1.5; margin: 0;">
                                                    <strong>Security Notice:</strong> If you didn't request this login, please ignore this email. 
                                                    This OTP will expire automatically in {OTP_EXPIRY_MINUTES} minutes.
                                                </p>
                                            </td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
                        </table>
                    </body>
                    </html>
                    """
                }
            ]
        }
        
        print(f"📧 Sending OTP to {email}...")
        result = mailjet.send.create(data=data)
        
        if result.status_code == 200:
            print(f"✅ OTP sent successfully to {email}")
            return True, "OTP sent successfully"
        else:
            error_msg = f"Failed to send OTP. Status: {result.status_code}"
            try:
                error_detail = result.json()
                error_msg += f" - {error_detail}"
            except:
                pass
            print(f"❌ {error_msg}")
            return False, error_msg
            
    except Exception as e:
        error_msg = f"Error sending OTP: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg

def is_valid_mailbox(email):
    """Check if email is in the allowed mailbox list"""
    mailbox_list = os.getenv('MAILBOX_LIST', '')
    
    if not mailbox_list or mailbox_list.strip() == '':
        print("⚠️ Warning: MAILBOX_LIST is empty in .env file")
        return False
    
    allowed_emails = [e.strip().lower() for e in mailbox_list.split(',') if e.strip()]
    
    if not allowed_emails:
        print("⚠️ Warning: No valid emails found in MAILBOX_LIST")
        return False
    
    is_valid = email.strip().lower() in allowed_emails
    
    if not is_valid:
        print(f"❌ Email {email} not in allowed list")
    
    return is_valid

def test_mailjet_connection():
    """Test Mailjet configuration without sending email"""
    is_valid, message = validate_mailjet_config()
    
    if is_valid:
        print(f"✅ {message}")
        api_key = os.getenv('MAILJET_API_KEY', 'NOT SET')
        print(f"   API Key: {api_key[:8]}..." if len(api_key) > 8 else f"   API Key: {api_key}")
        print(f"   Sender: {os.getenv('MAILJET_SENDER_EMAIL', 'NOT SET')}")
        mailbox_list = os.getenv('MAILBOX_LIST', 'NOT SET')
        print(f"   Allowed Emails: {mailbox_list}")
        return True
    else:
        print(f"❌ Mailjet Configuration Error: {message}")
        return False

# ---------------- Data helpers ----------------

def load_db():
    """Load processed table from PostgreSQL"""
    try:
        with get_db_connection() as conn:
            query = """
            SELECT message_id, mailbox, sender, receivers, cc, subject,
                   final_label, prob_neg, web_link, sender_domain,
                   processed_at, received_dt
            FROM processed
            ORDER BY received_dt DESC
            LIMIT 10000
            """
            df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"❌ Failed to read from PostgreSQL: {e}")
        return pd.DataFrame(columns=[
            "message_id", "mailbox", "sender", "receivers", "cc", "subject",
            "final_label", "prob_neg", "web_link", "sender_domain",
            "processed_at", "received_dt"
        ])
    
    if df.empty:
        return df
    
    if 'processed_at' in df.columns:
        df['processed_at'] = pd.to_datetime(df['processed_at'], errors='coerce')
    if 'received_dt' in df.columns:
        df['received_dt'] = pd.to_datetime(df['received_dt'], errors='coerce')
    
    df["client_domain"] = df.get("sender_domain",
                               df["sender"].apply(lambda s: (s or "").split("@")[-1].lower() if "@" in (s or "") else "")
                               ).fillna("unknown").astype(str)
    
    return df

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

def create_pie_chart(data, values, names, title):
    fig = px.pie(
        data, values=values, names=names, title=title, hole=0.4, color=names,
        color_discrete_map={'Negative': '#ff4444', 'Neutral': '#ffa500', 'Positive': '#00cc00'}
    )
    fig.update_traces(textposition='inside', textinfo='percent+label+value')
    fig.update_layout(height=400)
    return fig

# ---------------- Admin-driven helpers ----------------

def fetch_settings_from_db():
    """Read admin_settings and return dict key->value"""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT key, value FROM admin_settings", conn)
            return {k: v for k, v in zip(df['key'], df['value'])}
    except Exception as e:
        print("fetch_settings_from_db error:", e)
        return {}

def fetch_caution_words_from_db():
    """Return list[str] of caution words ordered newest-first"""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT word FROM caution_words ORDER BY created_at DESC", conn)
            return df['word'].astype(str).tolist() if not df.empty else []
    except Exception as e:
        print("fetch_caution_words_from_db error:", e)
        return []

# ---------------- Login Layout ----------------

def loginlayout():
    """Create login page layout"""
    return dbc.Container(
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H2("📧 Email Analytics Dashboard", className="text-center mb-4"),
                    html.Hr(),
                    
                    # Email Input Section
                    html.Div(id="email-input-section", children=[
                        html.H5("Login with Email OTP", className="mb-3"),
                        dbc.InputGroup([
                            dbc.InputGroupText("📧"),
                            dbc.Input(
                                id="login-email",
                                type="email",
                                placeholder="Enter your email address",
                                className="mb-3"
                            )
                        ]),
                        dbc.Button(
                            "Send OTP",
                            id="send-otp-button",
                            color="primary",
                            className="w-100 mb-3",
                            n_clicks=0
                        ),
                    ]),
                    
                    html.Div(id="email-status-message", className="text-center"),
                    
                    # OTP Input Section (hidden initially)
                    html.Div(id="otp-input-section", style={"display": "none"}, children=[
                        html.H5("Enter OTP", className="mb-3"),
                        dbc.InputGroup([
                            dbc.InputGroupText("🔐"),
                            dbc.Input(
                                id="login-otp",
                                type="text",
                                placeholder="Enter 6-digit OTP",
                                maxLength=6,
                                className="mb-3"
                            )
                        ]),
                        dbc.Button(
                            "Verify OTP",
                            id="verify-otp-button",
                            color="success",
                            className="w-100 mb-3",
                            n_clicks=0
                        ),
                    ]),
                    
                    html.Div(id="otp-status-message", className="text-center"),
                    
                    # Resend OTP Link
                    html.Div([
                        html.Small("Didn't receive OTP? ", className="text-muted"),
                        html.A("Resend", id="resend-otp-link", href="#", className="text-primary")
                    ], className="text-center"),
                    
                    html.Hr(),
                    
                    # Footer (REMOVED the allowed domains display)
                    html.Div([
                        html.Small("🔒 Protected by OTP authentication", className="text-muted")
                    ], className="text-center")
                    
                ], style={
                    "maxWidth": "400px",
                    "margin": "100px auto",
                    "padding": "2rem",
                    "backgroundColor": "white",
                    "borderRadius": "10px",
                    "boxShadow": "0 0 20px rgba(0,0,0,0.1)"
                })
            )
        ),
        fluid=True,
        style={
            "backgroundColor": "#f8f9fa",
            "minHeight": "100vh",
            "paddingTop": "50px"
        }
    )

# ---------------- Main Layout ----------------

def main_dashboard_layout():
    """Create main dashboard layout"""
    return dbc.Container([
        dcc.Store(id='data-store'),
        dcc.Store(id='table-sentiment-filter', data='all'),
        # NEW: stores for admin-driven settings and caution words
        dcc.Store(id='settings-store'),
        dcc.Store(id='caution-words-store'),
        dcc.Interval(id='auto-refresh', interval=AUTO_REFRESH_INTERVAL_SECONDS*1000, n_intervals=0),
        dcc.Location(id='url', refresh=False),
        dbc.Row([
            dbc.Col([
                html.H2("📊 Email Sentiment Analytics"),
                html.Hr(),
                html.Div([
                    html.H6("Navigation"),
                    dcc.RadioItems(
                        id='page-selector',
                        options=[
                            {'label': '🏠 Live Analysis', 'value': 'live'},
                            {'label': '📈 Trends & Transitions', 'value': 'trends'},
                            {'label': '📧 Email Details', 'value': 'emails'}
                        ],
                        value='live',
                        labelStyle={'display': 'block'}
                    ),
                ]),
                html.Hr(),
                html.Div([
                    html.H6("Settings"),
                    dbc.Checklist(
                        id='auto-refresh-toggle',
                        options=[{'label': '🔄 Auto-refresh', 'value': 'on'}],
                        value=[]
                    ),
                    dbc.Button("🔄 Refresh Now", id='btn-refresh', color='secondary',
                             className='mt-2', n_clicks=0)
                ], className='mb-3'),
                html.Div(id='sidebar-stats', className='text-muted mt-3'),
                dbc.Button("🚪 Logout", id='logout-button', color='outline-danger',
                          className='mt-3 w-100', n_clicks=0)
            ], width=3, style={'borderRight': '1px solid #ddd', 'minHeight': '100vh',
                               'paddingTop': '1rem'}),
            dbc.Col([
                html.Div(id='page-content', style={'padding': '1rem'})
            ], width=9)
        ])
    ], fluid=True)

# Root layout with session management
app.layout = html.Div([
    dcc.Store(id='session', storage_type='session'),
    dcc.Store(id='otp-storage', data={'otp': None, 'email': None, 'timestamp': None}),
    html.Div(id='root-content')
])

# ---------------- Callbacks ----------------

@app.callback(
    Output('root-content', 'children'),
    Input('session', 'data')
)
def render_root_content(session_data):
    """Render either login page or dashboard based on session"""
    if session_data and session_data.get('logged_in'):
        return main_dashboard_layout()
    else:
        return login_layout()

@app.callback(
    Output('session', 'data'),
    Input('logout-button', 'n_clicks'),
    State('session', 'data'),
    prevent_initial_call=True
)
def handle_logout(n_clicks, session_data):
    """Handle logout"""
    if n_clicks and n_clicks > 0:
        return {'logged_in': False}
    return session_data or {'logged_in': False}

@app.callback(
    Output('otp-storage', 'data'),
    Output('email-input-section', 'style'),
    Output('otp-input-section', 'style'),
    Output('email-status-message', 'children'),
    Input('send-otp-button', 'n_clicks'),
    State('login-email', 'value'),
    State('otp-storage', 'data'),
    prevent_initial_call=True
)
def send_otp(n_clicks, email, otp_storage):
    """Handle OTP sending"""
    if not email:
        return dash.no_update, dash.no_update, dash.no_update, dbc.Alert(
            "Please enter an email address", color="warning"
        )
    
    if not is_valid_mailbox(email):
        return dash.no_update, dash.no_update, dash.no_update, dbc.Alert(
            f"Email {email} not authorized. Please contact administrator.", 
            color="danger"
        )
    
    otp = generate_otp()
    success, message = send_otp_via_mailjet(email, otp)
    
    if success:
        new_otp_storage = {
            'otp': otp,
            'email': email,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        status_message = dbc.Alert(
            f"OTP sent successfully to {email}",
            color="success"
        )
        
        return new_otp_storage, {'display': 'none'}, {'display': 'block'}, status_message
    else:
        return dash.no_update, dash.no_update, dash.no_update, dbc.Alert(
            f"Failed to send OTP: {message}", color="danger"
        )

@app.callback(
    Output('session', 'data', allow_duplicate=True),
    Output('otp-status-message', 'children'),
    Input('verify-otp-button', 'n_clicks'),
    State('login-otp', 'value'),
    State('otp-storage', 'data'),
    State('session', 'data'),
    prevent_initial_call=True
)
def verify_otp(n_clicks, entered_otp, otp_storage, session_data):
    """Handle OTP verification"""
    if not entered_otp:
        return dash.no_update, dbc.Alert("Please enter the OTP", color="warning")
    
    if not otp_storage or not otp_storage.get('otp'):
        return dash.no_update, dbc.Alert("No OTP found. Please request a new one.", color="warning")
    
    otp_timestamp = datetime.fromisoformat(otp_storage['timestamp'])
    if datetime.utcnow() - otp_timestamp > timedelta(minutes=OTP_EXPIRY_MINUTES):
        return dash.no_update, dbc.Alert(
            "OTP expired. Please request a new one.", color="warning"
        )
    
    if entered_otp == otp_storage['otp']:
        new_session = {
            'logged_in': True,
            'email': otp_storage['email'],
            'login_time': datetime.utcnow().isoformat()
        }
        
        success_message = dbc.Alert(
            "Login successful! Redirecting...",
            color="success"
        )
        
        return new_session, success_message
    else:
        return dash.no_update, dbc.Alert("Invalid OTP. Please try again.", color="danger")

@app.callback(
    Output('send-otp-button', 'n_clicks'),
    Input('resend-otp-link', 'n_clicks'),
    prevent_initial_call=True
)
def resend_otp(n_clicks):
    """Handle OTP resend by triggering send button"""
    if n_clicks:
        return 1
    return 0

# ----------------- REFRESH: now writes settings + caution words -----------------
@app.callback(
    Output('data-store', 'data'),
    Output('settings-store', 'data'),
    Output('caution-words-store', 'data'),
    Input('btn-refresh', 'n_clicks'),
    Input('auto-refresh', 'n_intervals'),
    Input('auto-refresh-toggle', 'value'),
    prevent_initial_call=False
)
def refresh_data(n_clicks, n_intervals, auto_vals):
    triggered = dash.callback_context.triggered
    if triggered:
        trig_id = triggered[0]['prop_id']
        # if auto-refresh triggered but toggle is off, don't update
        if 'auto-refresh' in trig_id and (not auto_vals or 'on' not in auto_vals):
            raise dash.exceptions.PreventUpdate
    
    try:
        df = load_db()
        settings = fetch_settings_from_db()
        cautions = fetch_caution_words_from_db()
        # settings: dict, cautions: list[str]
        return df.to_json(date_format='iso', orient='split'), settings, cautions
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # return defaults
        return pd.DataFrame().to_json(date_format='iso', orient='split'), {}, []

# Update Interval interval (ms) when settings change
@app.callback(
    Output('auto-refresh', 'interval'),
    Input('settings-store', 'data'),
    prevent_initial_call=False
)
def update_interval(settings):
    try:
        if not settings:
            return AUTO_REFRESH_INTERVAL_SECONDS * 1000
        sec = int(settings.get('auto_refresh_interval', AUTO_REFRESH_INTERVAL_SECONDS))
        return max(5, sec) * 1000
    except Exception:
        return AUTO_REFRESH_INTERVAL_SECONDS * 1000

@app.callback(
    Output('sidebar-stats', 'children'),
    Input('data-store', 'data')
)
def update_sidebar_stats(data_json):
    if not data_json:
        return html.Div(["No data available"])
    
    df = pd.read_json(data_json, orient='split')
    total = len(df)
    now_ist = get_current_ist()
    last = df['processed_at'].max() if 'processed_at' in df.columns and len(df) > 0 else None
    
    if last and pd.notna(last):
        last_ist = pd.to_datetime(last) + IST_OFFSET
        last_str = last_ist.strftime('%Y-%m-%d %H:%M:%S')
    else:
        last_str = 'N/A'
    
    return html.Div([
        html.Div(f"Last updated: {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}"),
        html.Div(f"Total records: {total:,}"),
        html.Div(f"Last processed: {last_str}")
    ])

@app.callback(
    Output('page-content', 'children'),
    Input('page-selector', 'value'),
    Input('data-store', 'data')
)
def render_page(page, data_json):
    if not data_json:
        return dbc.Alert("No data – check the monitor and database.", color="warning")
    
    df = pd.read_json(data_json, orient='split')
    if df.empty:
        return dbc.Alert("No data available. Start the monitor to collect emails.", color="info")
    
    if page == 'live':
        return live_page_layout(df)
    if page == 'trends':
        return trends_layout(df)
    if page == 'emails':
        return emails_layout(df)
    
    return html.Div("Unknown page")

# ---------- Page builders (unchanged) ----------
def live_page_layout(df_all):
    mailboxes = ["All Mailboxes"] + sorted(df_all['mailbox'].dropna().unique().tolist())
    domain_counts = df_all['client_domain'].value_counts().reset_index()
    domain_counts.columns = ['client_domain', 'email_count']
    
    layout = html.Div([
        html.H3("📊 Live Sentiment Analysis"),
        dbc.Row([
            dbc.Col([
                html.Label("📬 Select Mailbox"),
                dcc.Dropdown(id='mailbox-selector', options=[{'label': m, 'value': m} for m in mailboxes], value='All Mailboxes', clearable=False)
            ], width=6),
            dbc.Col([
                html.Div(id='live-metrics')
            ], width=6)
        ], className='mb-3'),
        html.Hr(),
        dbc.Row([
            dbc.Col(dcc.Graph(id='overall-pie-chart'), width=4),
            dbc.Col(dcc.Graph(id='daily-pie-chart'), width=4),
            dbc.Col(dcc.Graph(id='activity-chart'), width=4)
        ]),
        html.Hr(),
        dcc.Tabs(id='live-page-tabs', value='domain-view', children=[
            dcc.Tab(label='👥 Domain View', value='domain-view', children=[
                html.Div([
                    html.H4("Filter by Client Domain"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Domain"),
                            dcc.Dropdown(
                                id='domain-view-selector',
                                options=[{'label': f"{row['client_domain']} ({row['email_count']} emails)", 'value': row['client_domain']}
                                        for _, row in domain_counts.iterrows()],
                                placeholder='Select a domain to filter emails...',
                                clearable=True
                            )
                        ], width=6)
                    ], className='mb-3'),
                    html.Div(id='domain-view-count-display', className='mb-2'),
                    html.Div(id='domain-view-table')
                ], style={'padding': '1rem'})
            ]),
            dcc.Tab(label='🔴 Live Feed', value='live-feed', children=[
                html.Div([
                    html.H4("🔴 Live Feed - All Emails"),
                    html.Div(id='sentiment-filter-info'),
                    dbc.Row([
                        dbc.Col(dbc.ButtonGroup([
                            dbc.Button("📊 Show All", id='table-btn-all', color='secondary', size='sm'),
                            dbc.Button("🔴 Negative Only", id='table-btn-negative', color='danger', size='sm'),
                            dbc.Button("🟡 Neutral Only", id='table-btn-neutral', color='warning', size='sm'),
                            dbc.Button("🟢 Positive Only", id='table-btn-positive', color='success', size='sm')
                        ]), width=6),
                        dbc.Col([
                            html.Label("Filter by Client Domain", className='me-2'),
                            dcc.Dropdown(id='live-client-filter', placeholder='Select client domain...', clearable=True)
                        ], width=6)
                    ], className='mb-2'),
                    html.Div(id='email-count-display', className='mb-2'),
                    html.Div(id='live-feed-table'),
                    dbc.Button("📥 Download CSV", id='download-btn', color='primary', className='mt-3'),
                    dcc.Download(id='download-dataframe-csv')
                ], style={'padding': '1rem'})
            ])
        ])
    ])
    
    return layout

def trends_layout(df):
    if len(df) > 0:
        min_date = to_ist(df['received_dt']).min().date()
        max_date = to_ist(df['received_dt']).max().date()
    else:
        min_date = get_current_ist().date()
        max_date = get_current_ist().date()
    
    domains = ["All Domains"] + sorted([d for d in df['client_domain'].unique() if d and d != "unknown"]) + (["unknown"] if "unknown" in df['client_domain'].values else [])
    
    layout = html.Div([
        html.H3("📈 Trends & Transitions Analysis"),
        dbc.Row([
            dbc.Col([
                html.Label("Date range"),
                dcc.DatePickerRange(id='date-range-picker', start_date=min_date, end_date=max_date, display_format='YYYY-MM-DD')
            ], width=4),
            dbc.Col([
                html.Label("Client Domain"),
                dcc.Dropdown(id='domain-filter', options=[{'label': d, 'value': d} for d in domains], value='All Domains', clearable=False)
            ], width=4),
            dbc.Col([
                html.Label("Sentiment"),
                dcc.Dropdown(id='sentiment-filter', options=[{'label':'Negative','value':'Negative'},{'label':'Neutral','value':'Neutral'},{'label':'Positive','value':'Positive'}], value=['Negative','Neutral','Positive'], multi=True)
            ], width=4)
        ], className='mb-3'),
        dbc.Row([
            dbc.Col(dcc.Graph(id='trends-pie-chart'), width=6)
        ], className='mb-4'),
        html.Hr(),
        html.H4("Client Sentiment Journey"),
        html.Div([
            html.Label("Select domain to analyze sentiment journey"),
            dcc.Dropdown(id='journey-domain-selector', options=[{'label': d, 'value': d} for d in domains if d!='All Domains'], placeholder="Select a domain...")
        ]),
        html.Div(id='journey-content')
    ])
    
    return layout

def emails_layout(df):
    mailboxes = ["All Mailboxes"] + sorted(df['mailbox'].dropna().unique().tolist())
    
    layout = html.Div([
        html.H3("📧 Detailed Email Records"),
        dbc.Row([
            dbc.Col([
                html.Label("Filter by Sentiment"),
                dcc.Dropdown(id='email-sentiment-filter', options=[{'label': 'Negative', 'value':'Negative'},{'label':'Neutral','value':'Neutral'},{'label':'Positive','value':'Positive'}], value=['Negative','Neutral','Positive'], multi=True)
            ], width=4),
            dbc.Col([
                html.Label("Filter by Mailbox"),
                dcc.Dropdown(id='email-mailbox-filter', options=[{'label': m, 'value': m} for m in mailboxes], value='All Mailboxes', clearable=False)
            ], width=4),
            dbc.Col([
                html.Label("Search"),
                dcc.Input(id='email-search-input', type='text', placeholder="Search...")
            ], width=4)
        ], className='mb-3'),
        html.Div(id='email-details-content')
    ])
    
    return layout
# ---------- Live Page Callbacks (with caution-words integration) ----------

@app.callback(
    Output('live-metrics', 'children'),
    Input('mailbox-selector', 'value'),
    Input('data-store', 'data')
)
def update_metrics(selected_mailbox, data_json):
    if not data_json:
        return ""
    
    df = pd.read_json(data_json, orient='split')
    if selected_mailbox != 'All Mailboxes':
        df = df[df['mailbox'] == selected_mailbox]
    
    total = len(df)
    neg = (df['final_label']=='Negative').sum()
    neu = (df['final_label']=='Neutral').sum()
    pos = (df['final_label']=='Positive').sum()
    avg_neg = df['prob_neg'].replace("",0).astype(float).mean() if total>0 else 0.0
    
    row = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("📧 Total Emails"), html.H4(f"{total:,}")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("🔴 Negative"), html.H4(f"{neg:,}"), html.Div(f"{(neg/total*100):.1f}%" if total>0 else "0%")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("🟡 Neutral"), html.H4(f"{neu:,}"), html.Div(f"{(neu/total*100):.1f}%" if total>0 else "0%")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("🟢 Positive"), html.H4(f"{pos:,}"), html.Div(f"{(pos/total*100):.1f}%" if total>0 else "0%")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("📈 Avg Neg Prob"), html.H4(f"{avg_neg:.3f}")])))
    ], className='g-2')
    
    return row

@app.callback(
    Output('overall-pie-chart', 'figure'),
    Output('daily-pie-chart', 'figure'),
    Output('activity-chart', 'figure'),
    Input('mailbox-selector', 'value'),
    Input('data-store', 'data')
)
def update_live_charts(selected_mailbox, data_json):
    if not data_json:
        return go.Figure(), go.Figure(), go.Figure()
    
    df = pd.read_json(data_json, orient='split')
    display = df if selected_mailbox=='All Mailboxes' else df[df['mailbox']==selected_mailbox]
    
    # Overall pie
    sent_counts = display['final_label'].value_counts().reset_index()
    sent_counts.columns = ['Sentiment','Count']
    fig_over = create_pie_chart(sent_counts, 'Count', 'Sentiment', f"Overall Sentiment - {selected_mailbox}")
    
    # Today's pie (in IST)
    today_ist = get_current_ist().date()
    display_ist = display.copy()
    display_ist['received_date_ist'] = to_ist(display_ist['received_dt']).dt.date
    df_today = display_ist[display_ist['received_date_ist'] == today_ist]
    
    if df_today.empty:
        empty = pd.DataFrame({'Sentiment':['Negative','Neutral','Positive'],'Count':[0,0,0]})
        fig_today = create_pie_chart(empty,'Count','Sentiment',"Today's Sentiment (0 emails)")
    else:
        dc = df_today['final_label'].value_counts().reset_index()
        dc.columns=['Sentiment','Count']
        fig_today = create_pie_chart(dc,'Count','Sentiment',f"Today's Sentiment ({len(df_today)} emails)")
    
    # Activity chart
    display['hour_ist'] = to_ist(display['processed_at']).dt.floor('h')
    hourly = display.groupby(['hour_ist','final_label']).size().reset_index(name='count').sort_values('hour_ist').tail(72)
    
    if hourly.empty:
        fig_act = go.Figure()
    else:
        fig_act = px.bar(hourly, x='hour_ist', y='count', color='final_label',
                        title="Hourly Email Distribution (Last 72 Hours)",
                        color_discrete_map={'Negative':'#ff4444','Neutral':'#ffa500','Positive':'#00cc00'})
        fig_act.update_layout(height=350, xaxis_title="Time (IST)")
    
    return fig_over, fig_today, fig_act

@app.callback(
    Output('table-sentiment-filter', 'data'),
    Output('sentiment-filter-info', 'children'),
    Input('table-btn-all', 'n_clicks'),
    Input('table-btn-negative', 'n_clicks'),
    Input('table-btn-neutral', 'n_clicks'),
    Input('table-btn-positive', 'n_clicks')
)
def update_sentiment_filter(all_clicks, neg_clicks, neu_clicks, pos_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'all', ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    info_messages = {
        'table-btn-all': "📊 Showing all emails",
        'table-btn-negative': "🔴 Showing only negative emails",
        'table-btn-neutral': "🟡 Showing only neutral emails",
        'table-btn-positive': "🟢 Showing only positive emails"
    }
    filter_map = {
        'table-btn-all': 'all',
        'table-btn-negative': 'Negative',
        'table-btn-neutral': 'Neutral',
        'table-btn-positive': 'Positive'
    }
    return filter_map.get(button_id, 'all'), info_messages.get(button_id, "")

@app.callback(
    Output('live-client-filter', 'options'),
    Input('data-store', 'data'),
    State('mailbox-selector', 'value')
)
def populate_live_client_filter(data_json, mailbox):
    if not data_json:
        return []
    
    df = pd.read_json(data_json, orient='split')
    if mailbox and mailbox != 'All Mailboxes':
        df = df[df['mailbox']==mailbox]
    
    domains = sorted([d for d in df['client_domain'].unique() if d and d != "unknown"])
    if "unknown" in df['client_domain'].values:
        domains.append("unknown")
    
    return [{'label': d, 'value': d} for d in domains]

# Update live table: now accepts caution words and highlights subjects
@app.callback(
    Output('live-feed-table', 'children'),
    Output('email-count-display', 'children'),
    Input('data-store','data'),
    Input('table-sentiment-filter', 'data'),
    Input('live-client-filter', 'value'),
    Input('caution-words-store', 'data'),   # NEW
    State('mailbox-selector','value')
)
def update_live_table(data_json, sentiment_filter, client_filter, caution_words, mailbox):
    if not data_json:
        return html.Div(), ""
    
    df = pd.read_json(data_json, orient='split')
    if mailbox and mailbox != 'All Mailboxes':
        df = df[df['mailbox']==mailbox]
    
    if sentiment_filter != 'all':
        df = df[df['final_label'] == sentiment_filter]
    
    if client_filter:
        df = df[df['client_domain'] == client_filter]
    
    if df.empty:
        return dbc.Alert("No emails match the selected filters", color="info"), "Showing 0 emails"
    
    disp = df[['processed_at', 'received_dt', 'mailbox', 'client_domain', 'sender',
               'receivers', 'cc', 'subject', 'final_label', 'web_link']].head(50).copy()
    disp['processed_at'] = format_ist(disp['processed_at'])
    disp['received_dt'] = format_ist(disp['received_dt'])
    
    # Apply caution-word highlighting if present
    subject_display = []
    if caution_words:
        import re
        pattern = "|".join([re.escape(w.lower()) for w in caution_words if w and w.strip()])
        if pattern:
            lower_subs = disp['subject'].fillna('').str.lower()
            is_caution = lower_subs.str.contains(pattern)
            for idx, row in disp.iterrows():
                subj = row['subject'] or ''
                if is_caution.loc[idx]:
                    subject_display.append("⚠️ " + subj)
                else:
                    subject_display.append(subj)
        else:
            subject_display = disp['subject'].fillna('').tolist()
    else:
        subject_display = disp['subject'].fillna('').tolist()
    
    disp['subject_html'] = [f"[{s}]({wl})" if wl and str(wl).strip() else s for s, wl in zip(subject_display, disp['web_link'])]
    
    table = dash_table.DataTable(
        id='live-feed-data-table',
        columns=[
            {"name": "Processed", "id": "processed_at"},
            {"name": "Received", "id": "received_dt"},
            {"name": "Mailbox", "id": "mailbox"},
            {"name": "Client", "id": "client_domain"},
            {"name": "Sender", "id": "sender"},
            {"name": "To", "id": "receivers"},
            {"name": "CC", "id": "cc"},
            {"name": "Subject", "id": "subject_html", "presentation": "markdown"},
            {"name": "Sentiment", "id": "final_label"}
        ],
        data=disp.to_dict('records'),
        page_size=12,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '6px'},
        style_data_conditional=[
            {'if': {'filter_query': '{final_label} = "Negative"'},
             'backgroundColor': '#fff0f0'}
        ]
    )
    
    return table, f"Showing {len(disp)} emails"

@app.callback(
    Output('download-dataframe-csv', 'data'),
    Input('download-btn', 'n_clicks'),
    State('data-store', 'data'),
    State('mailbox-selector', 'value'),
    State('table-sentiment-filter', 'data'),
    State('live-client-filter', 'value'),
    prevent_initial_call=True
)
def download_csv(n_clicks, data_json, mailbox, sentiment_filter, client_filter):
    if not data_json or n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    
    df = pd.read_json(data_json, orient='split')
    if mailbox and mailbox != 'All Mailboxes':
        df = df[df['mailbox'] == mailbox]
    
    if sentiment_filter != 'all':
        df = df[df['final_label'] == sentiment_filter]
    
    if client_filter:
        df = df[df['client_domain'] == client_filter]
    
    export_df = df.copy()
    export_df['processed_at_ist'] = format_ist(export_df['processed_at'])
    export_df['received_dt_ist'] = format_ist(export_df['received_dt'])
    csv_columns = ['processed_at_ist', 'received_dt_ist', 'mailbox', 'client_domain',
                   'sender', 'receivers', 'cc', 'subject', 'final_label', 'prob_neg']
    
    return dcc.send_data_frame(export_df[csv_columns].to_csv, "email_sentiment_export.csv")

@app.callback(
    Output('domain-view-selector', 'options'),
    Input('data-store', 'data'),
    State('mailbox-selector', 'value')
)
def populate_domain_view_selector(data_json, mailbox):
    if not data_json:
        return []
    
    df = pd.read_json(data_json, orient='split')
    if mailbox and mailbox != 'All Mailboxes':
        df = df[df['mailbox'] == mailbox]
    
    domain_counts = df['client_domain'].value_counts().reset_index()
    domain_counts.columns = ['client_domain', 'email_count']
    
    return [{'label': f"{row['client_domain']} ({row['email_count']} emails)", 'value': row['client_domain']}
            for _, row in domain_counts.iterrows()]

# Domain view table: also accept caution words to highlight subjects
@app.callback(
    Output('domain-view-table', 'children'),
    Output('domain-view-count-display', 'children'),
    Input('domain-view-selector', 'value'),
    Input('data-store', 'data'),
    Input('caution-words-store', 'data'),   # NEW
    State('mailbox-selector', 'value')
)
def update_domain_view_table(selected_domain, data_json, caution_words, mailbox):
    if not data_json:
        return html.Div(), ""
    
    df = pd.read_json(data_json, orient='split')
    if mailbox and mailbox != 'All Mailboxes':
        df = df[df['mailbox'] == mailbox]
    
    if not selected_domain:
        return html.Div(dbc.Alert("Select a domain to view emails", color="info")), ""
    
    filtered_df = df[df['client_domain'] == selected_domain]
    
    if filtered_df.empty:
        return html.Div(dbc.Alert(f"No emails found for domain: {selected_domain}", color="warning")), f"Showing 0 emails for {selected_domain}"
    
    disp = filtered_df[['processed_at', 'received_dt', 'mailbox', 'sender', 'receivers', 'cc', 'subject', 'final_label', 'web_link']].copy()
    disp['processed_at'] = format_ist(disp['processed_at'])
    disp['received_dt'] = format_ist(disp['received_dt'])
    
    # highlight caution words
    if caution_words:
        import re
        pattern = "|".join([re.escape(w.lower()) for w in caution_words if w and w.strip()])
        if pattern:
            lower_subs = disp['subject'].fillna('').str.lower()
            is_caution = lower_subs.str.contains(pattern)
            subject_display = []
            for idx, row in disp.iterrows():
                subj = row['subject'] or ''
                if is_caution.loc[idx]:
                    subject_display.append("⚠️ " + subj)
                else:
                    subject_display.append(subj)
            disp['subject_html'] = [f"[{s}]({wl})" if wl and str(wl).strip() else s for s, wl in zip(subject_display, disp['web_link'])]
        else:
            disp['subject_html'] = [f"[{s}]({wl})" if wl and str(wl).strip() else s for s, wl in zip(disp['subject'].fillna(''), disp['web_link'])]
    else:
        disp['subject_html'] = [f"[{s}]({wl})" if wl and str(wl).strip() else s for s, wl in zip(disp['subject'].fillna(''), disp['web_link'])]
    
    table = dash_table.DataTable(
        id='domain-view-data-table',
        columns=[
            {"name": "Processed At", "id": "processed_at"},
            {"name": "Received", "id": "received_dt"},
            {"name": "Mailbox", "id": "mailbox"},
            {"name": "Sender", "id": "sender"},
            {"name": "To", "id": "receivers"},
            {"name": "CC", "id": "cc"},
            {"name": "Subject", "id": "subject_html", "presentation": "markdown"},
            {"name": "Sentiment", "id": "final_label"}
        ],
        data=disp.to_dict('records'),
        page_size=15,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '6px'},
        style_data_conditional=[{'if': {'filter_query': '{final_label} = "Negative"'}, 'backgroundColor': '#fff0f0'}]
    )
    
    return table, f"Showing {len(disp)} emails for domain: {selected_domain}"

# ---------------- Trends & Journey callbacks remain unchanged ----------------
# (they use data only; no subject highlighting needed there)

# Email details: include caution words highlighting
@app.callback(
    Output('email-details-content','children'),
    Input('email-sentiment-filter','value'),
    Input('email-mailbox-filter','value'),
    Input('email-search-input','value'),
    Input('data-store','data'),
    Input('caution-words-store','data')   # NEW
)
def email_details(sentiments, mailbox, search, data_json, caution_words):
    if not data_json:
        return html.Div()
    
    df = pd.read_json(data_json, orient='split')
    detail_df = df.copy()
    
    if sentiments:
        detail_df = detail_df[detail_df['final_label'].isin(sentiments)]
    
    if mailbox and mailbox != 'All Mailboxes':
        detail_df = detail_df[detail_df['mailbox']==mailbox]
    
    if search and search.strip():
        q = search.lower()
        detail_df = detail_df[detail_df['subject'].fillna('').str.lower().str.contains(q) |
                             detail_df['sender'].fillna('').str.lower().str.contains(q) |
                             detail_df['receivers'].fillna('').str.lower().str.contains(q)]
    
    if detail_df.empty:
        return dbc.Alert("No emails match the current filters", color='info')
    
    disp = detail_df[['received_dt','client_domain','mailbox','sender','subject','final_label','prob_neg','web_link']].copy()
    disp['received_dt'] = format_ist(disp['received_dt'])
    
    # caution highlighting
    if caution_words:
        import re
        pattern = "|".join([re.escape(w.lower()) for w in caution_words if w and w.strip()])
        if pattern:
            lower_subs = disp['subject'].fillna('').str.lower()
            is_caution = lower_subs.str.contains(pattern)
            subject_display = []
            for idx, row in disp.iterrows():
                subj = row['subject'] or ''
                if is_caution.loc[idx]:
                    subject_display.append("⚠️ " + subj)
                else:
                    subject_display.append(subj)
            disp['subject_html'] = [f"[{s}]({wl})" if wl and str(wl).strip() else s for s, wl in zip(subject_display, disp['web_link'])]
        else:
            disp['subject_html'] = [f"[{s}]({wl})" if wl and str(wl).strip() else s for s, wl in zip(disp['subject'].fillna(''), disp['web_link'])]
    else:
        disp['subject_html'] = [f"[{s}]({wl})" if wl and str(wl).strip() else s for s, wl in zip(disp['subject'].fillna(''), disp['web_link'])]
    
    table = dash_table.DataTable(
        columns=[
            {"name":"Received","id":"received_dt"},
            {"name":"Client","id":"client_domain"},
            {"name":"Mailbox","id":"mailbox"},
            {"name":"From","id":"sender"},
            {"name":"Subject","id":"subject_html","presentation":"markdown"},
            {"name":"Sentiment","id":"final_label"},
            {"name":"Neg Prob","id":"prob_neg"}
        ],
        data=disp.to_dict('records'),
        page_size=15,
        style_table={'overflowX':'auto'},
        style_cell={'textAlign':'left','padding':'6px'},
        style_data_conditional=[{'if': {'filter_query': '{final_label} = "Negative"'}, 'backgroundColor':'#fff0f0'}]
    )
    
    csv_bytes = detail_df.to_csv(index=False).encode('utf-8')
    return html.Div([
        table,
        html.Br(),
        dcc.Download(id='download-details-csv'),
        dbc.Button("📥 Download Filtered Data (CSV)", id='download-details-btn', color='primary'),
        dcc.Store(id='email-detail-csv', data=csv_bytes.decode('utf-8'))
    ])

@app.callback(
    Output('download-details-csv','data'),
    Input('download-details-btn','n_clicks'),
    State('email-detail-csv','data'),
    prevent_initial_call=True
)
def download_details(n, csv_str):
    if not csv_str:
        raise dash.exceptions.PreventUpdate
    
    timestamp = get_current_ist().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(csv_str.encode('utf-8'), filename=f"email_sentiment_{timestamp}.csv")

# Test Mailjet configuration on startup
print("\n" + "="*60)
print("🚀 Starting Email Sentiment Analytics Dashboard")
print("="*60)
print("\n📧 Testing Mailjet Configuration...")
test_mailjet_connection()
print("="*60 + "\n")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    
    @app.server.route("/_health")
    def healthcheck():
        return "OK", 200
    
    print(f"🌐 Dashboard starting on http://0.0.0.0:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)