import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    y = df.drop(columns = ['id', 'message', 'genre', 'original'])
    
    y_pct = pd.DataFrame(y.sum(axis=0)/y.shape[0], columns = ['pct'])
    
    y_pct.sort_values(by = ['pct'], axis = 0, inplace = True, ascending = True)
    
    category_counts = [round(num, 2) * 100 for num in list(y_pct.pct)]
    category_names = list(y_pct.index)
   
    dist_data = y.sum(axis = 1)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x = category_counts,
                    y = category_names,
                    text = list(y_pct.index),
                    orientation = 'h',
                    width = 1,
                    opacity = 0.5
                )
            ],

            'layout': {
                'height' : '1024',
                'title': 'Appearance of Message Categories in the Training Dataset',
                'yaxis': {
                    'title' : {
                        'text' : 'Category',
                        'standoff' : '45'
                    },
                    'dtick' : '1',
                    'xanchor' : 'right',
                    'pad' : '25'
                },
                'xaxis': {
                    'title': "Percentage of Appearance in Train Dataset"
                },
                'plot_bgcolor' : 'rgb(255, 255, 255)',
                'legend' : {
                    'x' : '0.029',
                    'y' : '1.038', 
                    'font_size' : '10'
                },
                'margin' : {
                    'l' : '150', 
                    'r' : '20', 
                    't' : '70', 
                    'b' : '70'
                }
            }
        },
        
        {
            'data': [
                Histogram(
                    #x = category_counts,
                    x = dist_data,
                    #text = list(y_pct.index),
                    #orientation = 'h',
                    #width = 1,
                    opacity = 0.5,
                    xbins = dict(size = 1),
                    marker = dict(color = 'green', line = dict(color = 'black'))
                )
            ],

            'layout': {
                #'height' : '1024',
                'title': 'Distribution of the Number of Categories per Message in the Training Set',
                'yaxis': {
                    'title' : {
                        'text' : 'Number of Messages',
                        'standoff' : '45'
                    },
                    #'dtick' : '1',
                    'xanchor' : 'right',
                    'pad' : '25'
                },
                'xaxis': {
                    'title': "Number of Categories per Message"
                },
                'plot_bgcolor' : 'rgb(255, 255, 255)',
                'legend' : {
                    'x' : '0.029',
                    'y' : '1.038', 
                    'font_size' : '10'
                },
                'margin' : {
                    'l' : '150', 
                    'r' : '20', 
                    't' : '70', 
                    'b' : '70'
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()