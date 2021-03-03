import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_table
#Graphing
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#other apps
# from app import App, new_graph1, new_graph2
# import nav
import pandas as pd 
import numpy as np
import plotly.express as px
import pickle
import copy
import plotly.figure_factory as ff
#Flask
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import requests


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
# app.config.requests_pathname_prefix = ''
app.title = 'Churn Case Study Project'

app.config.suppress_callback_exceptions = True

#loading our data
with open('x_test.pickle','rb') as x_test_file:
    x_test = pickle.load(x_test_file)

with open('df_ids.pickle','rb') as df_ids_file:
    df_ids = pickle.load(df_ids_file)

ids_options = [{'label': str(x),'value': x} for x in list(x_test.index)]

model_options = [ #{'label':'XGBoostClassifier', 'value':'XGB'},
{'label':'Logistic Regression', 'value':'LOGIT'}, {'label':'RandomForestsClassifier', 'value':'RFC'}]

url = 'http://localhost:5000/' #'flask-app_1:5000'

#app callback function connecting to Flask App backend
@app.callback(Output('predictionText', 'children'),
              [Input('id_options', 'value'),
               Input('model_options', 'value')])
def update_prediction_text(id1, model):

    test1 =[list(np.array(x_test.loc[id1, :]))]

    # test1 = [list(pd.DataFrame(x_test.iloc[id1, :]).T)]

    # test1 =np.array(x_test.loc[id1, :])  #[list(  )]

    params ={'query':test1, 'model':model}   #test1 is the data to be passed 
    response = requests.get(url, json = params)
    return response.json()['prediction']


#app callback function to let user select new index value for customer
@app.callback(Output('probabilityText', 'children'),
              [Input('id_options', 'value'),
               Input('model_options', 'value')])
def update_probability_text(id1, model):

    print('ID is',id1,'Model is', model)

    test1 =[list(np.array(x_test.loc[id1, :]))]

    # test1 = list(pd.DataFrame(x_test.iloc[id1, :]).T)

    # np.array(x_test.loc[])    

    print('test1',test1)
    
    params ={'query':test1, 'model':model}   #test1 is the data to be passed 
    response = requests.get(url, json = params)

    if response.json()['prediction'] == 'Churn':
        return np.round(response.json()['confidence'][1],2)
    else:
        return np.round(response.json()['confidence'][0],2)


@app.callback(Output('age','children'),
             [Input('id_options', 'value')]) 
def output_stats(id1):

    num = x_test.loc[id1, 'Age']  #'$' + str('{:,. 2f}'.format(
    num = "{0:,.1f}".format(num)
    return str(num)

@app.callback(Output('num_products','children'),
             [Input('id_options', 'value')]) 
def output_stats(id1):

    num = x_test.loc[id1, 'NumOfProducts']  #'$' + str('{:,. 2f}'.format(
    num = "{0:,.0f}".format(num)
    return str(num)


@app.callback(Output('is_active','children'),
             [Input('id_options', 'value')]) 
def output_stats(id1):

    num = x_test.loc[id1, 'IsActiveMember']  #'$' + str('{:,. 2f}'.format(
    num = "{0:,.0f}".format(num)
    return str(num)


@app.callback(Output('balance','children'),
             [Input('id_options', 'value')]) 
def output_stats(id1):

    num = x_test.loc[id1, 'Balance']  #'$' + str('{:,. 2f}'.format(
    num = "{0:,.0f}".format(num)
    return '$' + str(num)


@app.callback(Output('gender','children'),
             [Input('id_options', 'value')]) 
def output_stats(id1):

    num = x_test.loc[id1, 'Gender']  #'$' + str('{:,. 2f}'.format(
    if num == 1:
    	result = 'Female'
    else:
    	result = 'Male'

    return result


#create layout
layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(
        l=30,
        r=30,
        b=20,
        t=40
    ),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation='h'),
    title='Satellite Overview',
)

app.layout = html.Div(
    [
        # html.Div(id = 'page-content'),
        #dcc.Store(id='aggregate_data'),
        html.Div(
            [
                html.Div(
                    [ 
                        html.Div([
                            html.H1([
                                'Churn Case Study: Real-time Customer Churn Predictions & Ranking Dashboard',
                                ],
                            ),
                        ], className = 'col-centered', style = {'padding-left':'150px'},
                        ),
                        html.Div([
                            html.Div([html.H5(
                                'by Philippe Heitzmann', 
                                style={'color': '#5dbcd2', 'font-style': 'italic', 'font-weight': 'bold', 'opacity': '0.8'}
                            )],
                            style = {'padding-left':'170px'},
                            ),
                            # html.Div([html.Img(
                            #     src = "https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTA3rgASWzdVLcpKDLzet7I-7a2FUGVtSRSqQ&usqp=CAU",
                            #     # className='two columns',
                            #     style = {'width':'35%'}
                            # )], #className = 'col-centered', 
                            # style = {'padding-left':'565px'},
                            # )
                        ], 
                        className = 'row',
                        )
                        
                    ],
                    className='eight columns',
                    style={'marginBottom': 10, 'marginTop': 0, 'width': '96%','padding-left':'0%', 
                    'padding-right':'0%',"border":"4px black solid", 'backgroundColor':'#FFF'}
                ),
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4('Real-time Machine Learning Predictions using Flask',
                            style = {'text-decoration':'underline','marginLeft':16}
                        ),
                        html.P("In order to output real-time loan default predictions for each of the models, I created a Flask app that takes pickle files of each of our models to output predictions for different subsets of our data. These different subsets are passed through as queries through the Flask API, which returns final predictions and predicted probabilities for each of our models.", 
                            style = {'marginLeft':7},
                            className="control_label"),
                    ]
                ),
                html.Div(
                    [
                        html.P("Select a Customer ID", 
                            style = {'marginLeft':15},
                            className="control_label"),
                        dcc.Dropdown(
                            id='id_options',
                            options=ids_options,
                            multi=False,
                            value=332,
                            className="dcc_control",
                            style = {'marginLeft':5}
                        ),
                        dcc.Checklist(
                            id='lock_selector',
                            # options=[
                            #     {'label': 'Lock camera', 'value': 'locked'}
                            # ],
                            value=[],
                            className="dcc_control"
                        ),
                        html.P("Select a Model to view",
                            style = {'marginLeft':15},
                            className="control_label"),

                        dcc.Dropdown(
                            id='model_options',
                            options=model_options,
                            multi=False,
                            value='RFC',
                            className="dcc_control",
                            style = {'marginLeft':5}
                        ),
                    ],
                ),
                html.Div(
                    [
                        html.P("Loan Default Prediction", style = {'color':'#FFF'}),
                        html.H6(
                            id="predictionText", className="info_text", style = {'color':'#FFF'}
                        )
                    ],
                    className='pretty_container four columns',
                    style={"border":"1px white solid", 'backgroundColor':'#3D95C1'}
                ),
                html.Div(
                    [
                        html.P("Prediction Confidence (/1.00)", style = {'color':'#FFF'}),
                        html.H6(
                            id="probabilityText", className="info_text", style = {'color':'#FFF'}
                        )
                    ],
                    className='pretty_container four columns',
                    style={"border":"1px white solid", 'backgroundColor':'#3D95C1'}
                ),
            ],
            className='row'
        ),

        html.Div(
            [
                html.H5("Customer Overview", 
                style = {'text-decoration':'underline','marginLeft':0})        
            ],
        ),

        html.Div([

            html.Div(
                    [
                        html.P("Age", style = {'color':'#FFF'}),
                        html.H6(
                            id="age", className="info_text", style = {'color':'#FFF'}
                        )
                    ],
                    className='pretty_container two columns',
                    style={"border":"1px white solid", 'backgroundColor':'#78B2D0'}
            ),

            html.Div(
                    [
                        html.P("Number of Products", style = {'color':'#FFF'}),
                        html.H6(
                            id="num_products", className="info_text", style = {'color':'#FFF'}
                        )
                    ],
                    className='pretty_container two columns',
                    style={"border":"1px white solid", 'backgroundColor':'#78B2D0'}
                ),

            html.Div(
                    [
                        html.P("Is Active", style = {'color':'#FFF'}),
                        html.H6(
                            id="is_active", className="info_text", style = {'color':'#FFF'}
                        )
                    ],
                    className='pretty_container two columns',
                    style={"border":"1px white solid", 'backgroundColor':'#78B2D0'}
                ),

            html.Div(
                    [
                        html.P("Balance", style = {'color':'#FFF'}),
                        html.H6(
                            id="balance", className="info_text", style = {'color':'#FFF'}
                        )
                    ],
                    className='pretty_container two columns',
                    style={"border":"1px white solid", 'backgroundColor':'#78B2D0'}
                ),

            html.Div(
                    [
                        html.P("Gender", style = {'color':'#FFF'}),
                        html.H6(
                            id="gender", className="info_text", style = {'color':'#FFF'}
                        )
                    ],
                    className='pretty_container two columns',
                    style={"border":"1px white solid", 'backgroundColor':'#78B2D0'}
                ),

            ],             
            className='row'
        ),
                html.Div(
            [
                html.H4(
                    'Ranking Highest Priority Customers',
                    style = {'text-decoration':'underline', 'marginLeft':0}
                ),
                html.P(
                    'The below interactive datatable allows a Chase Wealth Management advisor to rank and filter customers by their predicted probability of churning.',
                    style = {'marginLeft':0}
                ),
            ],
        ),
        html.Div([
                html.Div([
                    dash_table.DataTable(
                        # id='datatable-portfolio',
                        columns=[
                            {"name": i, "id": i, "deletable": True, "selectable": True} for i in df_ids.columns
                        ],
                        data=df_ids.to_dict('records'),
                        editable=True,
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        column_selectable="single",
                        row_selectable="multi",
                        row_deletable=True,
                        selected_columns=[],
                        selected_rows=[],
                        page_action="native",
                        page_current= 0,
                        page_size= 10,
                        style_cell={'fontSize':15, 'font-family':'sans-serif'},
                        ),
                    ], style = {'padding-left':'35px', 'padding-right':'0px'}
                    ),  
                ],
                className = 'row'
            ),

    ],
    id="mainContainer",
    style={
        "display": "flex",
        "flex-direction": "column"
    }
 )



if __name__ == '__main__':
    app.run_server(debug=True, host = '0.0.0.0', port = 80)
