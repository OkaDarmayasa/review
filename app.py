import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import regex as re
import string
import Sastrawi
import joblib
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

antonym_file_path = './Dataset/antonim_bahasa_indonesia.csv'
antonim = pd.read_csv(antonym_file_path)

def preprocess_pipeline(df, text_column):
    
    stop_factory = StopWordRemoverFactory()
    stopwords = stop_factory.get_stop_words()
    
    # Preprocess
    df['preprocessed'] = df[text_column].apply(preprocess)
    df["preprocessed"] = df["preprocessed"].str[2:]
    
    # Combine "nya" with previous word
    df['combined_nya'] = df['preprocessed'].apply(combine_nya_with_previous) 
    
    # Combine "di" with previous word
    df['combined_di'] = df['combined_nya'].apply(combine_di_with_next) 

    # Apply typo correction
    df['typo_corrected'] = df['combined_di'].apply(correct_typo)

    # Apply next word negation
    df['after_nwn_text'] = df['typo_corrected'].apply(next_word_negation)
    
    # Apply antonym swapping
    df['after_antonym_text'] = df['typo_corrected'].apply(swap_antonyms)
    
    # Remove stopwords on preprocessed only
    df["stopword_removed_processed"] = df["typo_corrected"].apply(
        lambda text: " ".join([word for word in text.split() if word not in stopwords])
    )
    
    # Remove stopwords on next word negation
    df["stopword_removed_nwn_processed"] = df["after_nwn_text"].apply(
        lambda text: " ".join([word for word in text.split() if word not in stopwords])
    )
    
    # Remove stopwords on antonym swapping
    df["stopword_removed_antonym_processed"] = df["after_antonym_text"].apply(
        lambda text: " ".join([word for word in text.split() if word not in stopwords])
    )
   
def preprocess(text):
    text1 = text.lower()   # case folding
    text4 = remove_emojis(text1)
    text5 = re.sub(r"\d+", "", text4)   # remove numbers
    text6 = text5.replace('\\n',' ')    # hapus karakter '\n'
    text7 = remove_punctuation(text6)
    result = text7.strip()   # remove whitespace
    return result

def remove_emojis(text):
    return str(text.encode('ascii', 'ignore'))

def remove_punctuation(text):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub(' ', text)

def combine_nya_with_previous(text):
    words = text.split()
    combined_text = []
    i = 0
    while i < len(words):
        if words[i] == 'nya' and i > 0:
            combined_text[-1] += 'nya'  # Combine "nya" with the previous word
        else:
            combined_text.append(words[i])
        i += 1
    return ' '.join(combined_text)

def combine_di_with_next(text):
    words = text.split()
    combined_text = []
    i = 0
    while i < len(words):
        if words[i] == 'di' and i < len(words) - 1:
            combined_text.append('di' + words[i+1])  # Combine "di" with the next word
            i += 1
        else:
            combined_text.append(words[i])
        i += 1
    return ' '.join(combined_text)

def correct_typo(text):
    corrections = pd.read_csv('./static/csv/typo_words.csv')

    corrections_dict = dict(zip(corrections['wrong'], corrections['right']))

    words = text.split()
    corrected_words = []
    
    for word in words:
        corrected_word = corrections_dict.get(word, word)
        corrected_words.append(corrected_word)

    corrected_text = ' '.join(corrected_words)
    
    return corrected_text

def next_word_negation(text):
    words = text.split()
    negation_words = ['tidak', 'bukan', 'belum', 'tak', 'kurang']

    new_words = []
    skip_next = False
    for i in range(len(words)):
        if skip_next:
            skip_next = False
            continue
        if words[i] in negation_words and i < len(words) - 1:
            new_words.append(words[i] + "_" + words[i + 1])
            skip_next = True
        else:
            new_words.append(words[i])

    return ' '.join(new_words)

def swap_antonyms(text):
    antonym_file_path = './static/csv/antonim_bahasa_indonesia.csv'
    antonim = pd.read_csv(antonym_file_path)
    antonim.head(10)

    words = text.split()
    negation_words = ['tidak', 'bukan', 'belum', 'tak', 'kurang']
    antonim_dict = dict(zip(antonim['word'], antonim['antonim']))

    new_words = []
    skip_next = False
    for i in range(len(words)):
        if skip_next:
            skip_next = False
            continue
        if words[i] in negation_words and i < len(words) - 1:
            next_word = words[i + 1]
            if next_word in antonim_dict:
                antonym = antonim_dict[next_word]
                skip_next = True
                new_words.append(antonym)
            else:
                new_words.append(words[i])
        else:
            new_words.append(words[i])

    return ' '.join(new_words)


st.sidebar.subheader('About the App')
st.sidebar.write('Text Classification App with Streamlit using a trained Naive Bayes model')
st.sidebar.write("This is just a small text classification app.")

model_path = './Model/nb_nwn_classifier.pkl' 
model = joblib.load(model_path)

vectorizer_path = './Model/nb_nwn_vectorizer.pkl'
vectorizer = joblib.load(vectorizer_path)

#start the user interface
st.title("Text Classification App")
st.write("Type in your text below and don't forget to press the enter button before clicking/pressing the 'Classify' button")

uploaded_file = st.file_uploader("Upload file with format .csv", type="csv")
my_text = st.text_input("Masukkan text yang ingin dicari sentimennya", max_chars=500, key='to_classify')

df = pd.DataFrame()

if my_text is not None:
    df1 = pd.DataFrame()
    df1['content'] = my_text

if uploaded_file is not None:
    # read csv
    df2 = pd.read_csv(uploaded_file)

df = pd.concat([df1, df2])

model_vectorizer_data = {
    'nb_preprocessed': ('nb_preprocessed_classifier.pkl', 'nb_preprocessed_vectorizer.pkl', 'stopword_removed_processed'),
    'nb_nwn': ('nb_nwn_classifier.pkl', 'nb_nwn_vectorizer.pkl', 'stopword_removed_nwn_processed'),
    'nb_antonym': ('nb_antonym_classifier.pkl', 'nb_antonym_vectorizer.pkl', 'stopword_removed_antonym_processed'),
    'svm_preprocessed': ('svm_preprocessed_classifier.pkl', 'svm_preprocessed_vectorizer.pkl', 'stopword_removed_processed'),
    'svm_nwn': ('svm_nwn_classifier.pkl', 'svm_nwn_vectorizer.pkl', 'stopword_removed_nwn_processed'),
    'svm_antonym': ('svm_antonym_classifier.pkl', 'svm_antonym_vectorizer.pkl', 'stopword_removed_antonym_processed')
}

selected_model = st.selectbox("Select a model", list(model_vectorizer_data.keys()))

if st.button('Classify', key='classify_button'):  
    processed_df = preprocess_pipeline(df, df.columns[0])

    # Load the selected model and vectorizer
    model_file, vectorizer_file, text_column = model_vectorizer_data[selected_model]
    model_path = './Model/' + model_file
    model = joblib.load(model_path)

    vectorizer_path = './Model/' + vectorizer_file
    vectorizer = joblib.load(vectorizer_path)

    X_new_vec = vectorizer.transform(processed_df[text_column])
    y_pred = model.predict(X_new_vec)

    # Create a DataFrame to display the results
    result_df = pd.DataFrame({
        'Text': processed_df['content'],
        'Processed Text': processed_df['stopword_removed_nwn_processed'],
        'Predicted Sentiment': y_pred
    })

    # Display the result DataFrame
    st.dataframe(result_df)
       
    