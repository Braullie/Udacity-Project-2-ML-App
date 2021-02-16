
## IMPORTANT

## Need to install the following library:
##  - scikit-multilearn

import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from skmultilearn.model_selection import iterative_train_test_split

from sqlalchemy import create_engine

import pickle

def load_data(database_filepath):
    """
    Load dataset from the database
    
    INPUT:
    database_filepath : Path of the database holding the dataset
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('dataset', con = engine)
    X = df.message
    y = df.drop(columns = ['id', 'message', 'genre', 'original'])
    categories = y.columns
    
    return X, y, categories
    

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Initialization of the model's pipeline and cross validation parameters
    """
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer())
            ])),

        # A Random Forest Classifier is used
        # The parameters are just used to initialize the object, could have left those values 
        # as default. The class_weight parameter is crucial to the performance of the model
        # due to classes being heavily imbalanced
        
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(
                max_depth = 5, 
                min_samples_leaf = 2, 
                n_estimators = 100,
                class_weight = "balanced"
            )
         )
        )
    ])
    
    parameters = {
        'clf__estimator__max_depth': [5, 10],
        'clf__estimator__min_samples_leaf': [2, 3],
        'clf__estimator__n_estimators': [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 10, scoring = 'f1_micro')
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluation of the fitted model on test data. Prints the model's performance on each category
    
    INPUT:
    model : Fitted model 
    X_test : Plain text messages used for testing
    Y_test : Categories belonging to the messages above
    category_names : Names of the 36 categories
    """
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns = category_names)

    Y_test = pd.DataFrame(Y_test, columns = category_names)
    
    cr_list = []

    for column in Y_test:
        cr = classification_report(
                Y_test[column], 
                Y_pred[column]
        )

        cr_list.append(cr)

        print(cr)


def save_model(model, model_filepath):
    """
    Save the model in the .pkl format
    
    INPUT:
    model : Fitted model
    model_filepath : Path where the model will be saved
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        # This part of the script is modified from the original in order to use
        # the iterative_train_test_split function from skmultilearn
        X = X.values
        X = X[:,np.newaxis]
        
        X_train, Y_train, X_test, Y_test = iterative_train_test_split(X, np.array(Y), test_size = 0.3)
        
        X_train = np.squeeze(X_train)
        X_test = np.squeeze(X_test)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()