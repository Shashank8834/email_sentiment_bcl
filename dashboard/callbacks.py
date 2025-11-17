# callbacks.py
# Additional callbacks for Plotly Dash Email Sentiment Analytics

from dash import Input, Output, State, html, dash_table, callback_context, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# Import app from the main module
# Note: This file should be imported at the END of app.py to avoid circular imports
from __main__ import app

# Callback for pie chart filter buttons
@app.callback(
    Output('clicked-sentiment-store', 'data'),
    [Input('btn-negative', 'n_clicks'),
     Input('btn-neutral', 'n_clicks'),
     Input('btn-positive', 'n_clicks'),
     Input('btn-all', 'n_clicks')]
)
def update_sentiment_filter(neg_clicks, neu_clicks, pos_clicks, all_clicks):
    """Update sentiment filter based on button clicks"""
    ctx = callback_context
    
    if not ctx.triggered:
        return None
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'btn-negative':
        return 'Negative'
    elif button_id == 'btn-neutral':
        return 'Neutral'
    elif button_id == 'btn-positive':
        return 'Positive'
    else:
        return None

# Callback for table filter buttons
@app.callback(
    [Output('live-feed-table', 'children'),
     Output('email-count-display', 'children'),
     Output('sentiment-filter-info', 'children')],
    [Input('table-btn-all', 'n_clicks'),
     Input('table-btn-negative', 'n_clicks'),
     Input('table-btn-neutral', 'n_clicks'),
     Input('table-btn-positive', 'n_clicks'),
     Input('clicked-sentiment-store', 'data'),
     Input('mailbox-selector', 'value'),
     Input('data-store', 'data')]
)
def update_live_feed_table(all_clicks, neg_clicks, neu_clicks, pos_clicks, 
                           pie_filter, selected_mailbox, data_json):
    """Update live feed table based on filters"""
    ctx = callback_context
    
    df = pd.read_json(data_json, orient='split')
    
    # Filter by mailbox
    if selected_mailbox != 'All Mailboxes':
        df_display = df[df['mailbox'] == selected_mailbox].copy()
    else:
        df_display = df.copy()
    
    # Determine active filter
    active_filter = None
    filter_info = None
    
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'table-btn-negative':
            active_filter = 'Negative'
        elif button_id == 'table-btn-neutral':
            active_filter = 'Neutral'
        elif button_id == 'table-btn-positive':
            active_filter = 'Positive'
        elif button_id == 'table-btn-all':
            active_filter = None
        elif button_id == 'clicked-sentiment-store' and pie_filter:
            active_filter = pie_filter
            filter_info = dbc.Alert(
                f"Showing {pie_filter} emails (filtered from pie chart)",
                color="info",
                dismissable=True,
                className="mb-3"
            )
    
    # Apply filter
    if active_filter:
        table_df = df_display[df_display['final_label'] == active_filter].copy()
    else:
        table_df = df_display.copy()
    
    # Create email count display
    count_display = html.P(f"Showing {len(table_df)} emails", className="fw-bold")
    
    # Format table data
    if not table_df.empty:
        # Format dates
        table_df['processed_at'] = pd.to_datetime(table_df['processed_at']).dt.strftime('%Y-%m-%d %H:%M')
        table_df['received_dt'] = pd.to_datetime(table_df['received_dt']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Select and rename columns
        display_df = table_df[[
            'processed_at', 'received_dt', 'mailbox', 'sender', 
            'receivers', 'cc', 'subject', 'final_label', 'web_link'
        ]].copy()
        
        display_df = display_df.rename(columns={
            'processed_at': 'Processed At',
            'received_dt': 'Received',
            'mailbox': 'Mailbox',
            'sender': 'Sender',
            'receivers': 'To',
            'cc': 'CC',
            'subject': 'Subject',
            'final_label': 'Sentiment'
        })
        
        # Create DataTable
        table = dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=[
                {'name': 'Processed At', 'id': 'Processed At'},
                {'name': 'Received', 'id': 'Received'},
                {'name': 'Mailbox', 'id': 'Mailbox'},
                {'name': 'Sender', 'id': 'Sender'},
                {'name': 'To', 'id': 'To'},
                {'name': 'CC', 'id': 'CC'},
                {'name': 'Subject', 'id': 'Subject', 'presentation': 'markdown'},
                {'name': 'Sentiment', 'id': 'Sentiment'}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{Sentiment} = "Negative"',
                    },
                    'backgroundColor': '#ffdddd'
                }
            ],
            page_size=20,
            sort_action='native',
            filter_action='native',
        )
        
        return table, count_display, filter_info
    else:
        return html.P("No emails to display", className="text-muted"), count_display, filter_info

# Callback for CSV download
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-btn", "n_clicks"),
    [State('mailbox-selector', 'value'),
     State('clicked-sentiment-store', 'data'),
     State('data-store', 'data')],
    prevent_initial_call=True
)
def download_csv(n_clicks, selected_mailbox, sentiment_filter, data_json):
    """Download filtered data as CSV"""
    df = pd.read_json(data_json, orient='split')
    
    # Apply filters
    if selected_mailbox != 'All Mailboxes':
        df = df[df['mailbox'] == selected_mailbox]
    
    if sentiment_filter:
        df = df[df['final_label'] == sentiment_filter]
    
    filename = f"emails_{selected_mailbox}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return dict(content=df.to_csv(index=False), filename=filename)

# Callback for Trends page
@app.callback(
    [Output('trends-pie-chart', 'figure'),
     Output('journey-content', 'children')],
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('domain-filter', 'value'),
     Input('sentiment-filter', 'value'),
     Input('journey-domain-selector', 'value'),
     Input('data-store', 'data')]
)
def update_trends_page(start_date, end_date, domain_filter, sentiment_filter, 
                       journey_domain, data_json):
    """Update trends and transitions page"""
    df = pd.read_json(data_json, orient='split')
    df_filtered = df.copy()
    
    # Apply filters
    if start_date and end_date:
        df_filtered = df_filtered[
            (pd.to_datetime(df_filtered['received_dt']).dt.date >= pd.to_datetime(start_date).date()) &
            (pd.to_datetime(df_filtered['received_dt']).dt.date <= pd.to_datetime(end_date).date())
        ]
    
    if domain_filter != 'All Domains':
        df_filtered = df_filtered[df_filtered['client_domain'] == domain_filter]
    
    if sentiment_filter:
        df_filtered = df_filtered[df_filtered['final_label'].isin(sentiment_filter)]
    
    # Pie chart
    sentiment_counts = df_filtered['final_label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig_pie = px.pie(
        sentiment_counts,
        values='Count',
        names='Sentiment',
        title="Sentiment Distribution (Filtered)",
        hole=0.4,
        color='Sentiment',
        color_discrete_map={'Negative': '#ff4444', 'Neutral': '#ffa500', 'Positive': '#00cc00'}
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label+value')
    fig_pie.update_layout(height=500)
    
    # Journey content
    journey_content = []
    
    if journey_domain:
        trans_df = df_filtered[df_filtered['client_domain'] == journey_domain].copy()
        trans_df = trans_df.sort_values('received_dt')
        
        if len(trans_df) > 1:
            # Create journey chart
            trans_df['email_number'] = range(1, len(trans_df) + 1)
            trans_df['date_short'] = pd.to_datetime(trans_df['received_dt']).dt.strftime('%m/%d %H:%M')
            
            sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
            trans_df['sentiment_value'] = trans_df['final_label'].map(sentiment_map)
            
            fig_journey = go.Figure()
            
            fig_journey.add_trace(go.Scatter(
                x=trans_df['email_number'],
                y=trans_df['sentiment_value'],
                mode='lines+markers',
                name='Sentiment Journey',
                line=dict(color='#888888', width=2),
                marker=dict(
                    size=16,
                    color=trans_df['final_label'].map({
                        'Negative': '#ff4444',
                        'Neutral': '#ffa500',
                        'Positive': '#00cc00'
                    }),
                    line=dict(width=2, color='white')
                ),
                text=trans_df.apply(lambda row: f"Email #{row['email_number']}<br>"
                                                  f"Date: {row['date_short']}<br>"
                                                  f"Sentiment: {row['final_label']}<br>"
                                                  f"Subject: {row['subject'][:50]}...", axis=1),
                hoverinfo='text'
            ))
            
            fig_journey.update_layout(
                title=f"Email Sentiment Sequence - {journey_domain}",
                xaxis_title="Email Sequence Number",
                yaxis_title="Sentiment",
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2],
                    ticktext=['Negative', 'Neutral', 'Positive'],
                    range=[-0.3, 2.3]
                ),
                height=450,
                hovermode='closest'
            )
            
            # Calculate metrics
            trans_df['prev_sentiment'] = trans_df['final_label'].shift(1)
            improvements = ((trans_df['prev_sentiment'] == 'Negative') & 
                          (trans_df['final_label'].isin(['Neutral', 'Positive']))).sum()
            deteriorations = ((trans_df['prev_sentiment'].isin(['Neutral', 'Positive'])) & 
                            (trans_df['final_label'] == 'Negative')).sum()
            stable = (trans_df['prev_sentiment'] == trans_df['final_label']).sum()
            
            journey_content = [
                html.H4(f"Sentiment Journey - {journey_domain}", className="mb-3"),
                dcc.Graph(figure=fig_journey),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("🔼 Improvements"),
                                html.H3(str(improvements)),
                                html.P("Times sentiment got better", className="text-muted")
                            ])
                        ])
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("🔽 Deteriorations"),
                                html.H3(str(deteriorations)),
                                html.P("Times sentiment got worse", className="text-muted")
                            ])
                        ])
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("➡️ Stable"),
                                html.H3(str(stable)),
                                html.P("Times sentiment stayed same", className="text-muted")
                            ])
                        ])
                    ], width=4)
                ], className="mt-3")
            ]
        else:
            journey_content = [
                dbc.Alert(f"Not enough data for domain: {journey_domain}. Need at least 2 emails.", 
                         color="info")
            ]
    
    return fig_pie, journey_content

# Callback for Client Analysis page
@app.callback(
    Output('client-analysis-content', 'children'),
    [Input('client-domain-selector', 'value'),
     Input('client-search-input', 'value'),
     Input('data-store', 'data')]
)
def update_client_analysis(selected_domain, search_query, data_json):
    """Update client analysis page"""
    if not selected_domain:
        return dbc.Alert("Please select a domain to analyze", color="info")
    
    df = pd.read_json(data_json, orient='split')
    domain_df = df[df['client_domain'] == selected_domain].copy()
    
    # Apply search filter
    if search_query:
        query_lower = search_query.lower()
        domain_df = domain_df[
            domain_df['subject'].fillna('').str.lower().str.contains(query_lower) |
            domain_df['sender'].fillna('').str.lower().str.contains(query_lower) |
            domain_df['receivers'].fillna('').str.lower().str.contains(query_lower)
        ]
    
    if domain_df.empty:
        return dbc.Alert(f"No emails found for domain: {selected_domain}", color="warning")
    
    # Metrics
    metrics = dbc.Row([
        dbc.Col([dbc.Card([dbc.CardBody([html.H6("Total Emails"), html.H4(len(domain_df))])])], width=3),
        dbc.Col([dbc.Card([dbc.CardBody([html.H6("Negative"), html.H4((domain_df['final_label'] == 'Negative').sum())])])], width=3),
        dbc.Col([dbc.Card([dbc.CardBody([html.H6("Neutral"), html.H4((domain_df['final_label'] == 'Neutral').sum())])])], width=3),
        dbc.Col([dbc.Card([dbc.CardBody([html.H6("Positive"), html.H4((domain_df['final_label'] == 'Positive').sum())])])], width=3),
    ], className="mb-4")
    
    # Pie chart
    domain_sentiment = domain_df['final_label'].value_counts().reset_index()
    domain_sentiment.columns = ['Sentiment', 'Count']
    fig_domain = px.pie(
        domain_sentiment,
        values='Count',
        names='Sentiment',
        title=f"Sentiment Distribution - {selected_domain}",
        hole=0.4,
        color='Sentiment',
        color_discrete_map={'Negative': '#ff4444', 'Neutral': '#ffa500', 'Positive': '#00cc00'}
    )
    
    # Probability distribution
    fig_prob = px.histogram(
        domain_df,
        x='prob_neg',
        nbins=20,
        title=f"Negative Probability Distribution - {selected_domain}",
        labels={'prob_neg': 'Negative Probability'}
    )
    
    return html.Div([
        metrics,
        html.Hr(),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_domain)], width=6),
            dbc.Col([dcc.Graph(figure=fig_prob)], width=6)
        ])
    ])

# Callback for Email Details page
@app.callback(
    Output('email-details-content', 'children'),
    [Input('email-sentiment-filter', 'value'),
     Input('email-mailbox-filter', 'value'),
     Input('email-search-input', 'value'),
     Input('data-store', 'data')]
)
def update_email_details(sentiment_filter, mailbox_filter, search_query, data_json):
    """Update email details page"""
    df = pd.read_json(data_json, orient='split')
    detail_df = df.copy()
    
    # Apply filters
    if sentiment_filter:
        detail_df = detail_df[detail_df['final_label'].isin(sentiment_filter)]
    
    if mailbox_filter != 'All Mailboxes':
        detail_df = detail_df[detail_df['mailbox'] == mailbox_filter]
    
    if search_query:
        query_lower = search_query.lower()
        detail_df = detail_df[
            detail_df['subject'].fillna('').str.lower().str.contains(query_lower) |
            detail_df['sender'].fillna('').str.lower().str.contains(query_lower) |
            detail_df['receivers'].fillna('').str.lower().str.contains(query_lower)
        ]
    
    if detail_df.empty:
        return dbc.Alert("No emails match the current filters", color="info")
    
    # Format table
    detail_df['received_dt'] = pd.to_datetime(detail_df['received_dt']).dt.strftime('%Y-%m-%d %H:%M')
    
    display_df = detail_df[[
        'received_dt', 'client_domain', 'mailbox', 'sender', 
        'subject', 'final_label', 'prob_neg'
    ]].copy()
    
    display_df = display_df.rename(columns={
        'received_dt': 'Received',
        'client_domain': 'Client',
        'mailbox': 'Mailbox',
        'sender': 'From',
        'subject': 'Subject',
        'final_label': 'Sentiment',
        'prob_neg': 'Neg Prob'
    })
    
    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in display_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'filter_query': '{Sentiment} = "Negative"'}, 'backgroundColor': '#ffdddd'}
        ],
        page_size=25,
        sort_action='native',
        filter_action='native',
    )
    
    return html.Div([
        html.P(f"Showing {len(detail_df)} emails", className="fw-bold mb-3"),
        table
    ])