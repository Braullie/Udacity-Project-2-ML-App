import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load both datasets used to create the final training dataset and merge them together
    
    INPUT:
    messages_filepath : Dataset containing the plain text messages
    categories_filepath : Dataset containing the categories associated with each message
    
    OUTPUT:
    df : Merged dataset containing the plain text messages and their corresponding categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(right = categories, on = 'id')
    
    return df

def clean_data(df):
    """
    Clean categories portion of the dataset in order to leave it ready to use in a 
    supervised ML model.
    
    INPUT:
    df : Dataset containing the plain text messages and their corresponding categories
    
    OUTPUT:
    df : Same dataset ready to be used in the ML model
    """
    categories = df.categories.str.split(pat = ';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x : x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x : x[-1:])

        # convert column from string to numeric
        categories[column] = categories[column].astype(float)
        
        # Remove columns with just 1 unique value (This is for the classification model)
        if len(categories[column].unique()) == 1:
            categories.drop(column, inplace = True, axis = 1)
        
    # drop the original categories column from `df`
    df.drop(columns = ['categories'], axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)

    # Remove rows where the related cell value is greater than or equal to 2
    # This is to be able to use a binary classification model on the related column
    df = df[df.related < 2.0]
    
    df = df[~df.duplicated(keep = 'first')]
    
    return df

def save_data(df, database_filename):
    """
    Save the dataset into the given SQL database
    
    INPUT:
    df : Dataset ready to be used in the ML model
    database_filename : SQL Database where the dataframe will be stored
    """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('dataset', engine, index=False)

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()