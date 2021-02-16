# Udacity-Project-2-ML-App
This project is created to host the files required for the second project of Udacity's Nanodegree Program.

The project is comprised of the following three elements:

## 1. ETL Pipeline: 

The main script is process_data.py which is contained in the 'data' folder and reads the two csv files (disaster_categories.csv, disaster_messages.csv), merges them into one and cleans the category columns to leave them ready for their use in the supervised ML model. The cleaning process consists of:
  - Remove unnecessary strings in each column in order to obtain the binary data indicating each category
  - Remove duplicate rows
  - Remove columns where there is only 1 distinct value
  - Remove rows of the 'related' column where it's value is equal to 2
After the cleaning is done, the dataset is saved in a SQLite database.

To use this pipeline, just change the csv file names andlocations in the process_data.py script to match the ones you're using, then run it.

## 1. ML Pipeline: 

### This step uses the library scikit-multilearn which can be installed with the following command
pip install scikit-multilearn

The main script is train_classifier.py contained in the 'models' folder which reads the stored dataset from the ETL Pipeline and fits a multioutput classification model in order to predict the different categories of each message. The process consists of:
  - Read the stored dataset
  - Create Train and Test datasets. This is done using the 'iterative_train_test_split' from the scikit-multilearn library due to sklearn's regular train_test_split not being optimal in giving a stratified train-test split in the multilabel scenario
  - Define the ML Pipeline
    - Define the NLP Pipeline using sklearn's CountVectorizer and TfidfTransformer
    - Define the ML model to be used. In this case, a Random Forest Classifier was used due to it's easiness to work with unbalanced datasets (just need to pass the class_weight = "balanced" parameter)
  - Fit the model using cross validation and choosing the best according to the 'f1_micro' metric
  - Evaluate the model using the Test datasets created in the previous steps. Precision, Recall and F1 score are given for each category.
  - Save the model as a .pkl file
  
Once saved, the model is ready to use in the app

To use this pipeline, install the scikit-multilearn library then change the SQLite database's name and location in the train_classifier.py script to match the one you're using, then run it.


## 1. App: 

The main script in run.py contained in the 'app' folder. It reads the model's .pkl file, the dataset from the SQLite database and uses them to create the visualizations and back end of the application. The application has two pages:
  - master.html: This is the main page users see when they launch the app. It has a box for users to write their queries and have them predicted by the model as well as two visualizations related to the training dataset, these visualizations are:
    - Count of the appearance of each category in the training dataset where it can be seen how unbalanced it really was and how some categories can't be well predicted due to their low overall appearance on the dataset
    - Histogram of the number of categories per message where it can be clearly seen that the mode is 0 thus saying that a great amount of the messages in the training dataset do not really fit even 1 category
  - go.html: This is the page that shows the predictions done by the model on the query input by the user in the master.html page. It highlights the categories that are predicted.
  
  To use this pipeline, install the scikit-multilearn library then change the SQLite database's name and location as well as the .pkl file name and location in the run.py script to match the ones you're using, then run it.
