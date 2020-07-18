import os

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

stock = pd.read_csv('IBM_Stock_Price.csv')
stock_train = pd.read_csv('IBM_Stock_Train.csv')
stock_test_pred = pd.read_csv('IBM_Stock_Test_And_Predicted.csv')
stock_vis = pd.read_csv('IBM_Stock_Month_Year_Date.csv')

server = app.server

app.layout = html.Div([
    html.H2('IBM Stock Price Prediction (Time Series)'),
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in ['Open/Close/High/Low Stock Prices', 'Average Price & Volume over Years', 'Average Price & Volume for each Months', 'Average Price & Volume for each days of month', 'Stock Prediction']],
        value='Stock Prediction'
    ),
    dcc.Graph(id="graph_with_selection"),
    html.P(id="slider_prediction")
])

@app.callback(dash.dependencies.Output('graph_with_selection', 'figure'),
              [dash.dependencies.Input('dropdown', 'value')])
def display_value(value):
    if value == "Open/Close/High/Low Stock Prices":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Open'],
                            mode='lines',
                            name='Open Stock Price'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Close'],
                            mode='lines',
                            name='Close Stock Price'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['High'],
                            mode='lines',
                            name='Close Stock Price'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Low'],
                            mode='lines',
                            name='Close Stock Price'))
        fig.update_layout(title='Open / Close / High / Low stock prices of IBM over years',
                   xaxis_title='Years',
                   yaxis_title='IBM Stock Prices')
        return fig
    elif value == "Average Price & Volume for each Months":
        
        month_group = stock_vis.groupby('month', as_index=False)[['Open', 'Volume']].mean()
        def toMonth(num):
            num = num-1
            month = ["January", "February", "March", "April", "May", "June", "July", "August",
                     "September", "October", "November", "December"]
            return month[num]
        month_group['month'] = month_group['month'].apply(toMonth)
        
        fig = make_subplots(rows=2, cols=1)
        fig.append_trace(go.Scatter(x=month_group['month'], y=month_group['Open'], 
                            mode='lines',
                            name='Average Open Stock Price') , row=1, col=1 )
        fig.append_trace(go.Scatter(x=month_group['month'], y=month_group['Volume'], 
                            mode='lines',
                            name='Average Volumes') , row=2, col=1 )
        fig.update_layout(height=900, title_text="Average Price & Volume for each Months")
        return fig
    elif value == "Average Price & Volume over Years":
        
        year_group = stock_vis.groupby('year', as_index=False)[['Open', 'Volume']].mean()
        
        fig = make_subplots(rows=2, cols=1)
        fig.append_trace(go.Scatter(x=year_group['year'], y=year_group['Open'], 
                            mode='lines',
                            name='Average Open Stock Price') , row=1, col=1 )
        fig.append_trace(go.Scatter(x=year_group['year'], y=year_group['Volume'], 
                            mode='lines',
                            name='Average Volumes') , row=2, col=1 )
        fig.update_layout(height=900, title_text="Average Price & Volume over Years")
        return fig
    elif value == "Average Price & Volume for each days of month":
        
        day_group = stock_vis.groupby('day', as_index=False)[['Open', 'Volume']].mean()
        
        fig = make_subplots(rows=2, cols=1)
        fig.append_trace(go.Scatter(x=day_group['day'], y=day_group['Open'], 
                            mode='lines',
                            name='Average Open Stock Price') , row=1, col=1 )
        fig.append_trace(go.Scatter(x=day_group['day'], y=day_group['Volume'], 
                            mode='lines',
                            name='Average Volumes') , row=2, col=1 )
        fig.update_layout(height=900, title_text="Average Price & Volume for each days of month")
        return fig
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_train['Date'], y=stock_train['Open'],
                            mode='lines',
                            name='Training Stock Price'))
        fig.add_trace(go.Scatter(x=stock_test_pred['Date'], y=stock_test_pred['Open'],
                            mode='lines',
                            name='Real Stock Price'))
        fig.add_trace(go.Scatter(x=stock_test_pred['Date'], y=stock_test_pred['predicted_price'],
                            mode='lines',
                            name='Predicted Stock Price'))
        fig.update_layout(height=900, title='Predicted Stock price of IBM',
                   xaxis_title='Years',
                   yaxis_title='IBM Stock Prices')
        return fig

if __name__ == '__main__':
    app.run_server(debug=True)