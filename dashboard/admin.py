# admin.py
# Admin/Backend dashboard for system configuration and monitoring

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Configuration
DB_PATH = os.environ.get("DB_PATH", "/data/monitor.db")
PORT = int(os.environ.get("PORT", 8051))  # admin dashboard default port if you run separately

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)

app.title = "Email Monitor - Admin Console"
server = app.server  # For compatibility

# Database functions
def load_db(max_rows=5000):
    """Load data from SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM processed ORDER BY processed_at DESC LIMIT ?",
            conn,
            params=(max_rows,),
            parse_dates=["processed_at", "received_dt"]
        )
    except Exception as e:
        conn.close()
        raise e
    conn.close()
    
    if not df.empty:
        df["client_domain"] = df.get("sender_domain", 
            df["sender"].apply(lambda s: (s or "").split("@")[-1].lower() if "@" in (s or "") else "")
        ).fillna("unknown").astype(str)
    
    return df

def get_db_stats():
    """Get database statistics"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    stats = {}
    
    try:
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
        print(f"Error getting stats: {e}")
        stats = {
            'total_records': 0,
            'first_processed': None,
            'last_processed': None,
            'unique_mailboxes': 0,
            'unique_senders': 0,
            'unique_domains': 0
        }
    finally:
        conn.close()
    
    return stats

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
            ])
        ], align="center")
    ], fluid=True),
    color="dark",
    dark=True,
    className="mb-4"
)

# Settings panel
settings_panel = dbc.Card([
    dbc.CardHeader("⚙️ System Settings"),
    dbc.CardBody([
        html.Label("Max Rows to Load", className="fw-bold"),
        dcc.Input(
            id='max-rows-input',
            type='number',
            value=5000,
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
            value=0.06,
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
        )
    ])
], className="mb-4")

# Layout
app.layout = html.Div([
    navbar,
    dcc.Store(id='admin-data-store'),
    dcc.Store(id='settings-store', data={'max_rows': 5000, 'neg_threshold': 0.06}),
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
            ], width=4),
            dbc.Col([
                html.Small(id='loaded-records-info', className="text-muted")
            ], width=4),
            dbc.Col([
                html.Small(f"💾 Database: {DB_PATH}", className="text-muted")
            ], width=4)
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
        return df.to_json(date_format='iso', orient='split')
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Update settings store
@app.callback(
    Output('settings-store', 'data'),
    [Input('max-rows-input', 'value'),
     Input('neg-threshold-slider', 'value')]
)
def update_settings(max_rows, neg_threshold):
    return {'max_rows': max_rows, 'neg_threshold': neg_threshold}

# Update footer
@app.callback(
    [Output('last-updated-time', 'children'),
     Output('loaded-records-info', 'children')],
    [Input('admin-data-store', 'data')]
)
def update_footer(data_json):
    if data_json:
        df = pd.read_json(data_json, orient='split')
        return (
            f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"📊 Loaded records: {len(df):,}"
        )
    return "🕐 No data", "📊 No records"

# Render tab content
@app.callback(
    Output('admin-tab-content', 'children'),
    [Input('admin-tabs', 'active_tab'),
     Input('admin-data-store', 'data'),
     Input('display-options', 'value')]
)
def render_tab_content(active_tab, data_json, display_options):
    if data_json is None:
        return dbc.Alert("Error loading data. Please check database connection.", color="danger")
    
    df = pd.read_json(data_json, orient='split')
    
    if df.empty:
        return dbc.Alert("No data available. Start the monitor to collect emails.", color="info")
    
    if active_tab == 'overview':
        return create_overview_tab(df)
    elif active_tab == 'database':
        return create_database_tab(df, display_options)
    elif active_tab == 'performance':
        return create_performance_tab(df)
    elif active_tab == 'config':
        return create_config_tab(display_options)
    
    return html.Div("Select a tab")

def create_overview_tab(df):
    """Create System Overview tab"""
    stats = get_db_stats()
    
    df['processed_at'] = pd.to_datetime(df['processed_at'])
    one_hour_ago = datetime.now() - timedelta(hours=1)
    recent = df[df['processed_at'] > one_hour_ago]
    processing_rate = len(recent)
    
    negative_count = (df["final_label"] == "Negative").sum()
    avg_neg_prob = df["prob_neg"].replace("", 0).astype(float).mean() if len(df) > 0 else 0.0
    
    metrics = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📧 Total Processed", className="text-muted"),
                    html.H3(f"{stats['total_records']:,}")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📬 Unique Mailboxes", className="text-muted"),
                    html.H3(f"{stats['unique_mailboxes']}")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("👤 Unique Senders", className="text-muted"),
                    html.H3(f"{stats['unique_senders']:,}")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🏢 Unique Domains", className="text-muted"),
                    html.H3(f"{stats['unique_domains']}")
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
    
    df['hour'] = df['processed_at'].dt.floor('H')
    hourly_counts = df.groupby('hour').size().reset_index(name='count')
    hourly_counts = hourly_counts.sort_values('hour').tail(24)
    
    fig_activity = px.line(
        hourly_counts,
        x='hour',
        y='count',
        title="Recent Processing Activity (Last 24 Hours)",
        labels={'hour': 'Time', 'count': 'Number of Emails'}
    )
    fig_activity.update_layout(height=300)
    
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
    
    return html.Div([
        html.H4("System Overview", className="mb-3"),
        metrics,
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
                    style_cell={'textAlign': 'left', 'padding': '10px'}
                )
            ], width=6)
        ])
    ])

def create_database_tab(df, display_options):
    """Create Database Management tab"""
    stats = get_db_stats()
    
    try:
        db_size = os.path.getsize(DB_PATH) / (1024 * 1024)
        db_size_str = f"{db_size:.2f} MB"
    except:
        db_size_str = "Unable to determine"
    
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
    
    content = [
        html.H4("Database Management", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Database Information"),
                    dbc.CardBody([
                        html.P([html.Strong("Database Path: "), html.Code(DB_PATH)]),
                        html.P([html.Strong("Total Records: "), f"{stats['total_records']:,}"]),
                        html.P([html.Strong("First Record: "), str(stats['first_processed'])]),
                        html.P([html.Strong("Last Record: "), str(stats['last_processed'])]),
                        html.P([html.Strong("Database Size: "), db_size_str])
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
    
    return html.Div(content)

def create_performance_tab(df):
    """Create Performance Metrics tab"""
    df['date'] = pd.to_datetime(df['processed_at']).dt.date
    daily_counts = df.groupby('date').size().reset_index(name='count')
    daily_counts = daily_counts.sort_values('date').tail(30)
    
    fig_timeline = px.line(
        daily_counts,
        x='date',
        y='count',
        title="Processing Timeline (Last 30 Days)",
        labels={'date': 'Date', 'count': 'Number of Emails'}
    )
    fig_timeline.update_layout(height=400)
    
    mailbox_counts = df['mailbox'].value_counts().head(10)
    
    fig_mailbox = px.bar(
        x=mailbox_counts.values,
        y=mailbox_counts.index,
        orientation='h',
        title="Top 10 Mailbox Activity",
        labels={'x': 'Number of Emails', 'y': 'Mailbox'}
    )
    fig_mailbox.update_layout(height=400)
    
    return html.Div([
        html.H4("Performance Metrics", className="mb-3"),
        dcc.Graph(figure=fig_timeline),
        html.Hr(),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_mailbox)], width=12)
        ])
    ])

def create_config_tab(display_options):
    """Create Configuration tab"""
    env_vars = {
        'CLIENT_ID': os.getenv('CLIENT_ID', 'Not set'),
        'TENANT_ID': os.getenv('TENANT_ID', 'Not set'),
        'DB_PATH': os.getenv('DB_PATH', 'Not set'),
        'POLL_INTERVAL': os.getenv('POLL_INTERVAL', 'Not set'),
        'MODEL_DIR': os.getenv('INFERENCE_MODEL_DIR', 'Not set'),
        'DEBUG_CC': os.getenv('DEBUG_CC', 'Not set')
    }
    
    masked_vars = {}
    for key, value in env_vars.items():
        if key in ['CLIENT_ID', 'CLIENT_SECRET']:
            masked_vars[key] = value[:8] + "..." if value != 'Not set' and len(value) > 8 else value
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
    
    return html.Div([
        html.H4("Configuration Management", className="mb-3"),
        html.Div(env_content)
    ])

if __name__ == '__main__':
    # Use the new app.run API (app.run_server is obsolete)
    app.run(debug=False, host='0.0.0.0', port=PORT)
