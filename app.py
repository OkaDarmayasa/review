import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import regex as re
import string
import Sastrawi
import joblib
import requests
import math
from collections import Counter, defaultdict
from io import BytesIO
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(layout="wide")

antonym_file_path = './Dataset/antonim_bahasa_indonesia.csv'
antonim = pd.read_csv(antonym_file_path)

def preprocess_pipeline(df):
    
    stop_factory = StopWordRemoverFactory()
    stopwords = stop_factory.get_stop_words()
    
    # Preprocess
    df['preprocessed'] = df.iloc[:, 0].apply(preprocess)
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
    return df
   
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
    corrections = pd.read_csv('./Dataset/typo_words.csv')

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
    antonym_file_path = './Dataset/antonim_bahasa_indonesia.csv'
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
                new_words.append(antonym.split(',')[0])
            else:
                new_words.append(words[i])
        else:
            new_words.append(words[i])

    return ' '.join(new_words)

def tf(text):
    words = text.split()
    tf_text = Counter(words)
    for i in tf_text:
        tf_text[i] = tf_text[i]/float(len(text))
    return tf_text

def idf(word, corpus):
    N = len(corpus)

    df = 0
    for i in corpus:
      if word in i:
        df += 1
    if df == 0:
        return 0
    idf = math.log10((1 + N) / (1 + df)) + 1
    return idf

def tfidf(corpus):
    documents_list = []
    for text in corpus:
        tf_idf_dictionary = {}
        computed_tf = tf(text)
        for word in computed_tf:
            tf_idf_dictionary[word] = computed_tf[word] * idf(word, corpus)
        documents_list.append(tf_idf_dictionary)
    return documents_list

def vectorize_tfidf(corpus, vocabulary=None):
    tfidf_result = tfidf(corpus)

    #if no vocab is assigned, meaning it is training the tf-idf model,
    #create new vocab
    if vocabulary is None:
        vocabulary = set()
        for tfidf_doc in tfidf_result:
            vocabulary.update(tfidf_doc.keys())
        vocabulary = sorted(list(vocabulary))

    #then create the tfidf matrix
                            #N        x    #len(d)
    tfidf_matrix = np.zeros((len(corpus), len(vocabulary)))
    for i, tfidf_doc in enumerate(tfidf_result):
        for j, term in enumerate(vocabulary):
            tfidf_matrix[i, j] = tfidf_doc.get(term, 0.0)

    return tfidf_matrix, vocabulary

def vectorize_tfidf_from_processed_text(processed_text_column, vocabulary=None):
    corpus = processed_text_column.tolist()
        #this is when fitting the tfidf
    if vocabulary is None:
        tfidf_matrix, vocabulary = vectorize_tfidf(corpus)
    else:
        #this is when transforming
        tfidf_matrix, _ = vectorize_tfidf(corpus, vocabulary=vocabulary)
    return tfidf_matrix, vocabulary

class MultinomialNaiveBayes:
    def __init__(self):
        self.class_priors = None
        self.conditional_probs = None

    def fit(self, X_train_tfidf, y_train):
        class_counts = np.bincount(y_train)
        self.class_priors = class_counts / len(y_train)

        self.conditional_probs = {}
        for label in np.unique(y_train):
            X_class = X_train_tfidf[y_train == label]
            total_word_counts = np.sum(X_class, axis=0)
            total_word_counts += 1
            total_words = np.sum(total_word_counts)
            self.conditional_probs[label] = total_word_counts / total_words

    def predict_proba(self, X_test_tfidf):
        scores = np.zeros((X_test_tfidf.shape[0], len(self.class_priors)))
        for label, conditional_probs in self.conditional_probs.items():
            log_probs = X_test_tfidf.dot(np.log(conditional_probs).T)
            scores[:, label] = log_probs.ravel()
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X_test_tfidf):
        return np.argmax(self.predict_proba(X_test_tfidf), axis=1)

def map_sentiment(sentiment):
    if sentiment == 0:
        return 'negative'
    elif sentiment == 1:
        return 'positive'
    else:
        return None 

def download_example_file(info):
    if info == "labelled":
        url = "https://github.com/OkaDarmayasa/review/raw/main/Dataset/test_labelled.csv"
    elif info == "unlabelled":
        url = "https://github.com/OkaDarmayasa/review/raw/main/Dataset/test_no_label.csv"
    response = requests.get(url)
    file_like_object = BytesIO(response.content)
    return file_like_object

def validate_page():
    st.title('Validate Model')
    st.write("Upload file CSV yang berisi data berlabel")
    st.write("Format CSV adalah 2 kolom dengan kolom kiri berupa text dan kolom kanan berupa label (0 untuk negatif atau 1 untuk positif)")

    with st.expander("Input data", expanded=True):  
        example_file = download_example_file("labelled")
        st.download_button(label="Download Example File", data=example_file, file_name='test_labelled.csv', mime='text/csv')

        uploaded_file = st.file_uploader("Upload file csv yang terdiri dari 2 kolom.", type="csv")

    default_data = [
                "barangnya gk bagus, jelek dipake",
                "Barang sesuai pesanan dan cepat sampai"
                ]
    default_data_sentiment = [0, 1]

    df = pd.DataFrame({'content': default_data, 'sentiment_label': default_data_sentiment})

    if uploaded_file is not None:
        # read csv
        csv_data = pd.read_csv(uploaded_file)
        df = pd.concat([df, csv_data], ignore_index=True)

    model_vectorizer_data = {
        'nb_preprocessed': ('nb_preprocessed_classifier.pkl', 'nb_preprocessed_vectorizer.pkl', 'stopword_removed_processed'),
        'nb_nwn': ('nb_nwn_classifier.pkl', 'nb_nwn_vectorizer.pkl', 'stopword_removed_nwn_processed'),
        'nb_antonym': ('nb_antonym_classifier.pkl', 'nb_antonym_vectorizer.pkl', 'stopword_removed_antonym_processed'),
        'svm_preprocessed': ('svm_preprocessed_classifier.pkl', 'svm_preprocessed_vectorizer.pkl', 'stopword_removed_processed'),
        'svm_nwn': ('svm_nwn_classifier.pkl', 'svm_nwn_vectorizer.pkl', 'stopword_removed_nwn_processed'),
        'svm_antonym': ('svm_antonym_classifier.pkl', 'svm_antonym_vectorizer.pkl', 'stopword_removed_antonym_processed')
    }

    if 'prev_selected_model' not in st.session_state:
        st.session_state.prev_selected_model = None

    selected_model = st.selectbox("Select a model", list(model_vectorizer_data.keys()))

    if selected_model != st.session_state.prev_selected_model or st.button('Classify', key='classify_button'):
        st.session_state.prev_selected_model = selected_model

        processed_df = preprocess_pipeline(df)
        y_test = df.iloc[:,1]

        model_file, vectorizer_file, text_column = model_vectorizer_data[selected_model]
        model_path = './Model/' + model_file
        model = joblib.load(model_path)

        vectorizer_path = './Model/' + vectorizer_file
        vocabulary = joblib.load(vectorizer_path)

        X_new_vec, _ = vectorize_tfidf_from_processed_text(processed_df[text_column], vocabulary=vocabulary)
        y_pred = model.predict(X_new_vec)

        accuracy = accuracy_score(y_test, y_pred)

        binary_y_pred = np.vectorize(map_sentiment)(y_pred)
        binary_y_test = np.vectorize(map_sentiment)(y_test)

        diff_sentiment_idx = np.where(binary_y_test != binary_y_pred)[0]
        
        result_df = pd.DataFrame({
            'Actual Sentiment': binary_y_test,
            'Predicted Sentiment': binary_y_pred,
            'Text': processed_df['content'],
            'Processed Text': processed_df['typo_corrected'],
            'Processed NWN Text': processed_df['after_nwn_text'],
            'Processed Antonym Text': processed_df['after_antonym_text']
        })

        filtered_df = result_df.iloc[diff_sentiment_idx]

        st.session_state.filtered_df = filtered_df
        st.session_state.result_df = result_df
        st.session_state.accuracy = accuracy


    if 'filtered_df' in st.session_state:
        # Add menu to select between viewing all data or only different sentiment rows
        show_all = st.radio("Show:", ("All Data", "Different Sentiments"))
        st.markdown(f"### Accuracy: {st.session_state.accuracy*100:.2f}%")
        if show_all == "All Data":
            with st.expander("Show Table", expanded=True):
                st.write(f"All Data: {len(st.session_state.result_df)} rows")
                st.write(f"Accurately Predicted: {len(st.session_state.result_df) - len(st.session_state.filtered_df)} rows")
                st.dataframe(st.session_state.result_df)
        elif show_all == "Different Sentiments":
            with st.expander("Show Table", expanded=True):
                st.write(f"Rows with Different Actual and Predicted Sentiments: {len(st.session_state.filtered_df)} rows")
                st.dataframe(st.session_state.filtered_df)
            
def test_page():
    st.title("Sentiment Prediction")
    with st.expander("Lihat detail model"):
        st.write("Ada 6 mesin yang di-deploy, yaitu 3 mesin naive bayes dan 3 mesin svm")
        st.write("Masing-masing mesin naive bayes dan svm terdiri dari mesin yang di-train pada dataset yang dilakukan 3 tahap preprocessing yang berbeda.")
        st.write("Yang pertama hanya dengan preprocessing saja.")
        st.write("Yang kedua dengan menggabungkan kata penanda negasi dengan kata selanjutnya menggunakan underscore (contoh: tidak_suka).")
        st.write("Yang ketiga dengan mengganti kata setelah kata penanda negasi dengan antonimnya (bila ada), contoh: tidak cepat => lambat.")
    
    with st.expander("Input data", expanded=True):    
        example_file = download_example_file("unlabelled")
        st.download_button(label="Download Example File", data=example_file, file_name='test_unlabelled.csv', mime='text/csv')
        
        uploaded_file = st.file_uploader("Upload file csv yang berisi text hanya pada 1 kolom yang sama.", type="csv")
        my_text = st.text_area("Masukkan text yang ingin dicari sentimennya", max_chars=2000, key='to_classify')

    default_data = [
                "barangnya gk bagus, jelek dipake",
                "Barang sesuai pesanan dan cepat sampai"
                ]

    df = pd.DataFrame({'content': default_data})

    if my_text is not None:
        my_text = my_text.splitlines()

        text_df = pd.DataFrame(my_text, columns=['content'])
        
        df = pd.concat([df, text_df], ignore_index=True)
    
    if uploaded_file is not None:
        # read csv
        csv_data = pd.read_csv(uploaded_file)
        df = pd.concat([df, csv_data], ignore_index=True)

    model_vectorizer_data = {
        'nb_preprocessed': ('nb_preprocessed_classifier.pkl', 'nb_preprocessed_vectorizer.pkl', 'stopword_removed_processed'),
        'nb_nwn': ('nb_nwn_classifier.pkl', 'nb_nwn_vectorizer.pkl', 'stopword_removed_nwn_processed'),
        'nb_antonym': ('nb_antonym_classifier.pkl', 'nb_antonym_vectorizer.pkl', 'stopword_removed_antonym_processed'),
        'svm_preprocessed': ('svm_preprocessed_classifier.pkl', 'svm_preprocessed_vectorizer.pkl', 'stopword_removed_processed'),
        'svm_nwn': ('svm_nwn_classifier.pkl', 'svm_nwn_vectorizer.pkl', 'stopword_removed_nwn_processed'),
        'svm_antonym': ('svm_antonym_classifier.pkl', 'svm_antonym_vectorizer.pkl', 'stopword_removed_antonym_processed')
    }

    if 'prev_selected_model' not in st.session_state:
        st.session_state.prev_selected_model = None

    selected_model = st.selectbox("Select a model", list(model_vectorizer_data.keys()))

    if selected_model != st.session_state.prev_selected_model or st.button('Classify', key='classify_button'):
        st.session_state.prev_selected_model = selected_model
        processed_df = preprocess_pipeline(df)
        model_file, vectorizer_file, text_column = model_vectorizer_data[selected_model]
        model_path = './Model/' + model_file
        model = joblib.load(model_path)

        vectorizer_path = './Model/' + vectorizer_file
        vocabulary = joblib.load(vectorizer_path)

        X_new_vec, _ = vectorize_tfidf_from_processed_text(processed_df[text_column], vocabulary=vocabulary)
        y_pred = model.predict(X_new_vec)

        binary_y_pred = np.vectorize(map_sentiment)(y_pred)
        result_df = pd.DataFrame({
            'Predicted Sentiment': binary_y_pred,
            'Content': processed_df['content'],
            'Processed Text': processed_df['typo_corrected'],
            'Processed NWN Text': processed_df['after_nwn_text'],
            'Processed Antonym Text': processed_df['after_antonym_text']
        })
        with st.expander("Show Table", expanded=True):
            st.dataframe(result_df)


def main():
    st.title("Negation Handling Text Classification App")

    # Sidebar menu
    choice = st.sidebar.radio("Select:", ("Validate Model", "Test Model"))

    # Display page based on user choice
    if choice == "Validate Model":
        validate_page()
    elif choice == "Test Model":
        test_page()

if __name__ == "__main__":
    main()