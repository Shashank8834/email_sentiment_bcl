# dashboard/app.py
# Dash conversion of your Streamlit app_dashboard.py

# Add these imports at the top of app.py

import os
from datetime import datetime, timedelta
import sqlite3
import io

import pandas as pd
import numpy as np

import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

from dotenv import load_dotenv
load_dotenv()

# Config
DB_PATH = os.environ.get("DB_PATH", "/data/monitor.db")
AUTO_REFRESH_INTERVAL_SECONDS = 30

# IST offset from UTC
IST_OFFSET = timedelta(hours=5, minutes=30)

# Initialize app
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server
app.title = "Email Sentiment Analytics"

# ---------------- Data helpers ----------------
def load_db(path=DB_PATH):
    """Load processed table from sqlite and return a cleaned dataframe."""
    try:
        conn = sqlite3.connect(path)
        df = pd.read_sql_query("SELECT * FROM processed ORDER BY received_dt DESC", conn)
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Failed to read DB at {path}: {e}")
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "message_id", "mailbox", "sender", "receivers", "cc", "subject",
            "final_label", "prob_neg", "web_link", "sender_domain", "processed_at", "received_dt"
        ])
    # Ensure required columns exist
    required_cols = ["message_id", "mailbox", "sender", "receivers", "cc", "subject",
                     "final_label", "prob_neg", "web_link", "sender_domain", "processed_at", "received_dt"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""
    
    # Convert datetime strings to datetime objects (stored as UTC in DB)
    if 'processed_at' in df.columns:
        df['processed_at'] = pd.to_datetime(df['processed_at'], errors='coerce')
    if 'received_dt' in df.columns:
        df['received_dt'] = pd.to_datetime(df['received_dt'], errors='coerce')
    
    df["client_domain"] = df.get("sender_domain",
        df["sender"].apply(lambda s: (s or "").split("@")[-1].lower() if "@" in (s or "") else "")
    ).fillna("unknown").astype(str)
    return df

def to_ist(dt_series):
    """Convert UTC datetime series to IST for display."""
    if dt_series is None:
        return dt_series
    # Add IST offset to UTC times
    return pd.to_datetime(dt_series) + IST_OFFSET

def format_ist(dt_series, format_str='%Y-%m-%d %H:%M IST'):
    """Format datetime series as IST string."""
    ist_times = to_ist(dt_series)
    return ist_times.dt.strftime(format_str)

def get_current_ist():
    """Get current time in IST."""
    return datetime.utcnow() + IST_OFFSET

def create_pie_chart(data, values, names, title):
    fig = px.pie(
        data,
        values=values,
        names=names,
        title=title,
        hole=0.4,
        color=names,
        color_discrete_map={'Negative': '#ff4444', 'Neutral': '#ffa500', 'Positive': '#00cc00'}
    )
    fig.update_traces(textposition='inside', textinfo='percent+label+value')
    fig.update_layout(height=400)
    return fig

def create_timeline_chart(df, domain=None):
    timeline_df = df.copy()
    if domain:
        timeline_df = timeline_df[timeline_df['client_domain'] == domain]
    if timeline_df.empty:
        return go.Figure()
    timeline_df['date'] = to_ist(timeline_df['received_dt']).dt.date
    daily = timeline_df.groupby(['date', 'final_label']).size().reset_index(name='count')
    fig = px.line(
        daily, x='date', y='count', color='final_label',
        title=f"Sentiment Trend Over Time{' - ' + domain if domain else ''}",
        color_discrete_map={'Negative': '#ff4444', 'Neutral': '#ffa500', 'Positive': '#00cc00'}
    )
    fig.update_layout(height=350, xaxis_title="Date", yaxis_title="Number of Emails")
    return fig

def create_domain_comparison_chart(df):
    if df.empty:
        return go.Figure()
    domain_sentiment = df.groupby(['client_domain', 'final_label']).size().reset_index(name='count')
    fig = px.bar(
        domain_sentiment, x='client_domain', y='count', color='final_label',
        title="Sentiment Distribution by Client Domain", barmode='group',
        color_discrete_map={'Negative': '#ff4444', 'Neutral': '#ffa500', 'Positive': '#00cc00'}
    )
    fig.update_layout(height=400, xaxis_title="Client Domain", yaxis_title="Email Count")
    return fig

# ---------------- Layout ----------------
app.layout = dbc.Container([
    dcc.Store(id='data-store'),
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
                dbc.Button("🔄 Refresh Now", id='btn-refresh', color='secondary', className='mt-2', n_clicks=0)
            ], className='mb-3'),
            html.Div(id='sidebar-stats', className='text-muted mt-3')
        ], width=3, style={'borderRight': '1px solid #ddd', 'minHeight': '100vh', 'paddingTop': '1rem'}),
        dbc.Col([
            html.Div(id='page-content', style={'padding': '1rem'})
        ], width=9)
    ])
], fluid=True)


# ---------------- Callbacks ----------------

# Load data into store on refresh, interval, or manual click
@app.callback(
    Output('data-store', 'data'),
    Input('btn-refresh', 'n_clicks'),
    Input('auto-refresh', 'n_intervals'),
    Input('auto-refresh-toggle', 'value'),
    prevent_initial_call=False
)
def refresh_data(n_clicks, n_intervals, auto_vals):
    # If auto-refresh is off, only refresh on manual click (btn-refresh)
    triggered = dash.callback_context.triggered
    # If auto-refresh disabled and the trigger is interval, don't reload
    if triggered:
        trig_id = triggered[0]['prop_id']
        if 'auto-refresh' in trig_id and (not auto_vals or 'on' not in auto_vals):
            raise dash.exceptions.PreventUpdate
    try:
        df = load_db(DB_PATH)
        return df.to_json(date_format='iso', orient='split')
    except Exception as e:
        print(f"Error loading data: {e}")
        # store an empty DataFrame with error flag
        return pd.DataFrame().to_json(date_format='iso', orient='split')

# Sidebar stats
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

# Render selected page
@app.callback(
    Output('page-content', 'children'),
    Input('page-selector', 'value'),
    Input('data-store', 'data')
)
def render_page(page, data_json):
    if not data_json:
        return dbc.Alert("No data — check the monitor and DB file.", color="warning")
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

# ---------- Page builders ----------
def live_page_layout(df_all):
    mailboxes = ["All Mailboxes"] + sorted(df_all['mailbox'].dropna().unique().tolist())
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
    ])
    return layout

def trends_layout(df):
    if len(df) > 0:
        min_date = to_ist(df['received_dt']).min().date()
        max_date = to_ist(df['received_dt']).max().date()
    else:
        min_date = get_current_ist().date()
        max_date = get_current_ist().date()
    
    domains = ["All Domains"] + sorted([d for d in df['client_domain'].unique() if d and d != "unknown"]) + ["unknown"]
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

# ---------- Live page callbacks ----------
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
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("📈 Avg Neg Prob"), html.H4(f"{avg_neg:.3f}")]))),
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

# Populate client domain dropdown
@app.callback(
    Output('live-client-filter', 'options'),
    Input('data-store', 'data'),
    State('mailbox-selector', 'value')
)
def populate_client_filter(data_json, mailbox):
    if not data_json:
        return []
    df = pd.read_json(data_json, orient='split')
    if mailbox and mailbox != 'All Mailboxes':
        df = df[df['mailbox']==mailbox]
    
    domains = sorted([d for d in df['client_domain'].unique() if d and d != "unknown"])
    if "unknown" in df['client_domain'].unique():
        domains.append("unknown")
    
    return [{'label': d, 'value': d} for d in domains]

# Live feed table and quick filter handlers
@app.callback(
    Output('live-feed-table', 'children'),
    Output('email-count-display', 'children'),
    Input('table-btn-all','n_clicks'),
    Input('table-btn-negative','n_clicks'),
    Input('table-btn-neutral','n_clicks'),
    Input('table-btn-positive','n_clicks'),
    Input('live-client-filter', 'value'),
    Input('data-store','data'),
    State('mailbox-selector','value')
)
def update_live_table(n_all, n_neg, n_neu, n_pos, client_domain, data_json, mailbox):
    ctx = dash.callback_context
    if not data_json:
        return html.Div(), ""
    df = pd.read_json(data_json, orient='split')
    if mailbox and mailbox != 'All Mailboxes':
        df = df[df['mailbox']==mailbox]
    
    # Filter by client domain
    if client_domain:
        df = df[df['client_domain']==client_domain]
    
    # determine which button triggered
    trig = None
    if ctx.triggered:
        trig = ctx.triggered[0]['prop_id'].split('.')[0]
    if trig == 'table-btn-negative':
        df = df[df['final_label']=='Negative']
    elif trig == 'table-btn-neutral':
        df = df[df['final_label']=='Neutral']
    elif trig == 'table-btn-positive':
        df = df[df['final_label']=='Positive']
    
    # prepare table
    if df.empty:
        filter_msg = f" for client '{client_domain}'" if client_domain else ""
        return html.Div(dbc.Alert(f"No emails to display{filter_msg}", color="info")), f"Showing 0 emails"
    
    disp = df[['processed_at','received_dt','mailbox','client_domain','sender','receivers','cc','subject','final_label','web_link']].copy()
    disp['processed_at'] = format_ist(disp['processed_at'])
    disp['received_dt'] = format_ist(disp['received_dt'])
    
    # create clickable subject column
    disp['subject_html'] = disp.apply(lambda r: f"[{r['subject']}]({r['web_link']})" if pd.notna(r['web_link']) and str(r['web_link']).strip() else r['subject'], axis=1)
    
    table = dash_table.DataTable(
        id='live-table',
        columns=[
            {"name":"Processed At","id":"processed_at"},
            {"name":"Received","id":"received_dt"},
            {"name":"Mailbox","id":"mailbox"},
            {"name":"Client","id":"client_domain"},
            {"name":"Sender","id":"sender"},
            {"name":"To","id":"receivers"},
            {"name":"CC","id":"cc"},
            {"name":"Subject","id":"subject_html","presentation":"markdown"},
            {"name":"Sentiment","id":"final_label"}
        ],
        data=disp.to_dict('records'),
        page_size=12,
        style_table={'overflowX':'auto'},
        style_cell={'textAlign':'left','padding':'6px'},
        style_data_conditional=[
            {
                'if': {'filter_query': '{final_label} = "Negative"'},
                'backgroundColor': '#fff0f0'
            }
        ]
    )
    return table, f"Showing {len(disp)} emails"

# CSV download from live table
@app.callback(
    Output('download-dataframe-csv', 'data'),
    Input('download-btn', 'n_clicks'),
    State('data-store', 'data'),
    State('mailbox-selector','value'),
    prevent_initial_call=True
)
def download_csv(n_clicks, data_json, mailbox):
    if not data_json:
        raise dash.exceptions.PreventUpdate
    df = pd.read_json(data_json, orient='split')
    if mailbox and mailbox!='All Mailboxes':
        df = df[df['mailbox']==mailbox]
    csv_string = df.to_csv(index=False)
    timestamp = get_current_ist().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(csv_string.encode('utf-8'), filename=f"emails_{mailbox}_{timestamp}.csv")

# ---------- Trends callbacks ----------
@app.callback(
    Output('trends-pie-chart','figure'),
    Input('data-store','data'),
    Input('date-range-picker','start_date'),
    Input('date-range-picker','end_date'),
    Input('domain-filter','value'),
    Input('sentiment-filter','value')
)
def update_trends_pie(data_json, start_date, end_date, domain, sentiments):
    if not data_json:
        return go.Figure()
    df = pd.read_json(data_json, orient='split')
    
    # Convert received_dt to IST for filtering
    df['received_dt_ist'] = to_ist(df['received_dt'])
    
    if start_date and end_date:
        start_date_obj = pd.to_datetime(start_date).date()
        end_date_obj = pd.to_datetime(end_date).date()
        df = df[(df['received_dt_ist'].dt.date >= start_date_obj) & (df['received_dt_ist'].dt.date <= end_date_obj)]
    
    if domain and domain!='All Domains':
        df = df[df['client_domain']==domain]
    if sentiments:
        df = df[df['final_label'].isin(sentiments)]
    
    sent_counts = df['final_label'].value_counts().reset_index()
    sent_counts.columns = ['Sentiment','Count']
    return create_pie_chart(sent_counts, 'Count', 'Sentiment', "Sentiment Distribution (Filtered)")

@app.callback(
    Output('journey-content','children'),
    Input('data-store','data'),
    Input('journey-domain-selector','value')
)
def build_journey(data_json, domain):
    if not data_json or not domain:
        return html.Div()
    df = pd.read_json(data_json, orient='split')
    trans_df = df[df['client_domain']==domain].sort_values('received_dt')
    
    if len(trans_df) <= 1:
        return dbc.Alert(f"Not enough data for domain: {domain}", color='info')
    
    # Convert to IST for display
    trans_df['date_time_ist'] = to_ist(trans_df['received_dt'])
    trans_df['date_short'] = trans_df['date_time_ist'].dt.strftime('%m/%d %H:%M')
    
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    trans_df['sentiment_value'] = trans_df['final_label'].map(sentiment_map)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trans_df['date_time_ist'],
        y=trans_df['sentiment_value'],
        mode='lines+markers',
        marker=dict(size=12, color=trans_df['final_label'].map({'Negative':'#ff4444','Neutral':'#ffa500','Positive':'#00cc00'})),
        text=trans_df.apply(lambda r: f"{r['date_short']}<br>{r['final_label']}<br>{r['subject'][:120]}", axis=1),
        hoverinfo='text'
    ))
    fig.update_layout(
        title=f"Email Sentiment Sequence - {domain}", 
        xaxis_title="Date (IST)",
        yaxis=dict(tickmode='array', tickvals=[0,1,2], ticktext=['Negative','Neutral','Positive']), 
        height=450
    )
    
    # summary stats
    trans_df['prev_sentiment'] = trans_df['final_label'].shift(1)
    improvements = ((trans_df['prev_sentiment']=='Negative') & (trans_df['final_label'].isin(['Neutral','Positive']))).sum()
    deteriorations = ((trans_df['prev_sentiment'].isin(['Neutral','Positive'])) & (trans_df['final_label']=='Negative')).sum()
    stable = (trans_df['prev_sentiment'] == trans_df['final_label']).sum()
    
    stats = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("🔼 Improvements"), html.H4(str(improvements))])), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("🔽 Deteriorations"), html.H4(str(deteriorations))])), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("➡️ Stable"), html.H4(str(stable))])), width=4)
    ], className='my-3')
    
    return html.Div([dcc.Graph(figure=fig), stats])

# ---------- Emails callbacks ----------
@app.callback(
    Output('email-details-content','children'),
    Input('email-sentiment-filter','value'),
    Input('email-mailbox-filter','value'),
    Input('email-search-input','value'),
    Input('data-store','data')
)
def email_details(sentiments, mailbox, search, data_json):
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
    disp['subject_html'] = disp.apply(lambda r: f"[{r['subject']}]({r['web_link']})" if pd.notna(r['web_link']) and str(r['web_link']).strip() else r['subject'], axis=1)
    
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

# Download handler for details
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

# Run app
if __name__ == '__main__':
    # default port 8050; change via env if needed
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host='0.0.0.0', port=port)