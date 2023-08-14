from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer

#####
import csv
import requests
import io  # Import the io module for working with bytes streams

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# model=pickle.load(open('ff_model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template('open_ended.html')

# @app.route('/')
# def index():
#     return render_template('csv.html', textarea_content="")

# working - but takes column
# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'csv_file' in request.files:
#         csv_file = request.files['csv_file']
#         if csv_file.filename != '':
#             csv_stream = io.StringIO(
#                 csv_file.stream.read().decode("UTF-8"), newline=None)
#             csv_reader = csv.reader(csv_stream, delimiter=',')

#             # Read the first row of the CSV (assuming it contains the column headers)
#             # Use [] as default if no header row
#             header_row = next(csv_reader, [])

#             # Initialize a list to store textarea content
#             textarea_content = []

#             # Iterate over each column header and append the corresponding cell value
#             for header in header_row:
#                 # Use [""] as default if no cell value
#                 cell_value = next(csv_reader, [""])[0]
#                 textarea_content.append(cell_value)

#             return render_template('csv.html', textarea_content=textarea_content)
#     return "No CSV file uploaded."

# working


@app.route('/upload', methods=['POST'])
def upload():
    if 'csv_file' in request.files:
        csv_file = request.files['csv_file']
        if csv_file.filename != '':
            csv_stream = io.StringIO(
                csv_file.stream.read().decode("UTF-8"), newline=None)
            csv_reader = csv.reader(csv_stream, delimiter=',')

            # Skip the first row, which contains the column headers
            next(csv_reader, [])

            # Read the first row of the CSV (assuming it contains the data)
            # Use [] as default if no header row
            data_row = next(csv_reader, [])

            # Initialize a list to store textarea content
            textarea_content = []

            # Iterate over each data in the first row
            for datum in data_row:
                # Add the datum to the textarea content list
                textarea_content.append(datum)

            return render_template('open_ended.html', textarea_content=textarea_content)
    return "No CSV file uploaded."


# @app.route('/predict_opn',methods=['POST','GET'])
# def predict():
#     text = 'I actively pursue new experiences and take calculated risks both personally and professionally. I believe in stepping out of my comfort zone to grow and learn. Embracing challenges has strengthened my adaptability, problem-solving, and innovation abilities, allowing me to contribute valuable perspectives to the teams I work with.'

#     # Convert to lowercase
#     text = text.lower()

#     # Remove non-word and non-whitespace characters
#     text = re.sub(r'[^\w\s]', '', text)

#     # Remove digits
#     text = re.sub(r'\d', '', text)

#     tokens = word_tokenize(text)
#     # Initialize lemmatizer
#     lemmatizer = WordNetLemmatizer()
#     stop_words = set(stopwords.words('english'))
#     # Lemmatize tokens
#     lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
#     return lemmatized_tokens
# Define the list of words you want to count
opn_target_words = ['new', 'risk', 'try', 'believe', 'learn', 'opportunity',
                    'potential', 'experience', 'challenge', 'growth', 'look', 'explore', 'enjoy', 'seek']

csn_target_words = ['work', 'time', 'ensure', 'task',
                    'complete', 'accurately', 'deadline', 'prioritize']

ext_target_words = ['connection', 'people', 'relationship', 'social',
                    'meeting', 'building', 'experience', 'opportunity', 'professional']

agr_target_words = ['feedback', 'believe', 'approach', 'understand',
                    'ask', 'person', 'listen', 'learn', 'opportunity', 'help']

neu_target_words = ['problem', 'stress',
                    'difficult', 'challenge', 'stressful', 'break']

# Define the predict route


@app.route('/predict_opn', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # q1 - openness
        opn_text = request.form['opn']

        # Perform text processing (tokenization, lemmatization, etc.)
        opn_lemmatized_tokens = preprocess_text(opn_text)

        # Count occurrences of common words in lemmatized_tokens
        opn_word_count = sum(
            1 for word in opn_lemmatized_tokens if word in opn_target_words)

        opn_score = opn_word_count/len(opn_target_words)*5
        opn_score = round(opn_score, 2)  # Round off to 2 decimal places

        # q2 - concientiousness
        csn_text = request.form['csn']

        # Perform text processing (tokenization, lemmatization, etc.)
        csn_lemmatized_tokens = preprocess_text(csn_text)

        # Count occurrences of common words in lemmatized_tokens
        csn_word_count = sum(
            1 for word in csn_lemmatized_tokens if word in csn_target_words)

        csn_score = csn_word_count/len(csn_target_words)*5
        csn_score = round(csn_score, 2)  # Round off to 2 decimal places

        # q3 - extraversion
        ext_text = request.form['ext']

        # Perform text processing (tokenization, lemmatization, etc.)
        ext_lemmatized_tokens = preprocess_text(ext_text)

        # Count occurrences of common words in lemmatized_tokens
        ext_word_count = sum(
            1 for word in ext_lemmatized_tokens if word in ext_target_words)

        ext_score = ext_word_count/len(ext_target_words)*5
        ext_score = round(ext_score, 2)  # Round off to 2 decimal places

        # q4 - agreeableness
        agr_text = request.form['agr']

        # Perform text processing (tokenization, lemmatization, etc.)
        agr_lemmatized_tokens = preprocess_text(agr_text)

        # Count occurrences of common words in lemmatized_tokens
        agr_word_count = sum(
            1 for word in agr_lemmatized_tokens if word in agr_target_words)

        agr_score = agr_word_count/len(agr_target_words)*5
        agr_score = round(agr_score, 2)  # Round off to 2 decimal places

        # q5 - neuroticism
        neu_text = request.form['neu']

        # Perform text processing (tokenization, lemmatization, etc.)
        neu_lemmatized_tokens = preprocess_text(neu_text)

        # Count occurrences of common words in lemmatized_tokens
        neu_word_count = sum(
            1 for word in neu_lemmatized_tokens if word in neu_target_words)

        neu_score = neu_word_count/len(neu_target_words)*5
        neu_score = round(neu_score, 2)  # Round off to 2 decimal places

        # Final Score
        opn_ended_score = (opn_score + csn_score +
                           ext_score + agr_score + neu_score)/5
        # Round off to 2 decimal places
        opn_ended_score = round(opn_ended_score, 2)

        # Pass the processed data to the template
        return render_template('open_ended.html', opn_lemmatized_tokens=opn_lemmatized_tokens, csn_lemmatized_tokens=csn_lemmatized_tokens, ext_lemmatized_tokens=ext_lemmatized_tokens, agr_lemmatized_tokens=agr_lemmatized_tokens, neu_lemmatized_tokens=neu_lemmatized_tokens, opn_score=opn_score, csn_score=csn_score, ext_score=ext_score, agr_score=agr_score, neu_score=neu_score, opn_ended_score=opn_ended_score)

    # Return the template for GET requests
    return render_template('open_ended.html')


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove non-word and non-whitespace characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove digits
    text = re.sub(r'\d', '', text)

    tokens = word_tokenize(text)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words('english'))

    # Lemmatize tokens
    lemmatized_tokens = [lemmatizer.lemmatize(
        word) for word in tokens if word.lower() not in stop_words]

    return lemmatized_tokens


if __name__ == '__main__':
    app.run(debug=True)
