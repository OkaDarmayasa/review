# Link to the Web App
https://review-sa793yyjutnsgzmvgwpqfv.streamlit.app/

# About the Model
There are 2 types of machine learning classifiers used in this:
1. Naive Bayes
2. Support Vector Machine

There are 3 different preprocessing steps being done which create 3 different models for each classifier, which are:
1. Only with preprocessing
2. Preprocessing + Next Word Negation, which is combining a negation word with the word after it with an underscore (_). Exp: tidak suka => tidak_suka
3. Preprocessing + swapping with antonym, which works by swapping a negation word and one word after it with the antonym of the word after only if the word after has an antonym.

In total, there are 6 different models created. In the web app, you can choose which model to use to classify the text.

# How to Use the Web App
I think, the web app will not always be up. But when it is up, you can have 2 different types of input:
1. File upload.
2. Text box typing.

For file upload, the requirements is kinda strict, which are:
1. CSV file
2. Only has 1 column
3. The top most row must be called 'content' because that is the name of the header of the dataframe.

For the box typing, you can type multiple instance/row of text and the system will recognized a different row if there is a new line being inputted in the text box.

After inputting the text by file or text box, the next step is to click 'Classify'. After you have clicked 'Classify' at least once, you can now automatically change the result of the classifier whenever you change the model without having to click 'Classify' anymore.

The result that is displayed is a table that contains the predicted sentiment (positive or negative), the raw text, and a bunch of preprocessed text of different preprocessing steps.

# How to Remake
##Remaking the Model
The main code is in the Alur Utama.ipynb file. In there, most file/data are inputted through Google Drive, so make sure to upload all of the folders in this repo into your Google Drive folder and copy the path of each file accordingly. 

###The Naive Bayes Classifier
The Naive Bayes Classifier code that is being used here is from scikit-learn MultinomialNB. 
###The Support Vector Machine Classifier
The Support Vector Machine Classifier code that is being used here is from scikit-learn SVC.
###The Term-Frequency Inverse-Document-Frequency (TF-IDF) Feature Extraction 
The TF-IDF code that is being used here is from scikit-learn TfidfVectorizer.
##Remaking the Web App
There are 2 ways to remake the web app:
1. Locally.
   You have to install streamlit in your device.
   More can be seen here: docs.streamlit.io/get-started/installation/command-line#create-an-environment-using-venv
2. Online
   You have to login to streamlit.io with your Google or Github account and then setup a Github repository.
   More can be seen here: docs.streamlit.io/streamlit-community-cloud/get-started#get-started
