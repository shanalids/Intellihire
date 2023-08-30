from flask import Flask, request, url_for, redirect, render_template, render_template_string
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

import csv
import requests
import io  # Import the io module for working with bytes streams

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# model=pickle.load(open('ff_model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template('Personality_prediction/requirement.html', textarea_content="", slider_values="")


@app.route('/responses')
def go_to_responses():
    return render_template('Personality_prediction/responses.html', textarea_content="", slider_values="")


@app.route('/upload_responses', methods=['POST'])
def upload_selfrate():
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

            # Initialize lists to store slider values and textarea content
            slider_values = []
            textarea_content = []

            # Iterate over each data in the first 25 columns and convert them to integers
            for i in range(25):
                if i < len(data_row):
                    datum = data_row[i]
                    # Convert the datum to an integer and append it to the slider values list
                    slider_values.append(int(datum))

            # Iterate over the next 5 columns for textarea content
            for i in range(25, 30):
                if i < len(data_row):
                    datum = data_row[i]
                    # Append the datum to the textarea content list
                    textarea_content.append(datum)

            return render_template('Personality_prediction/responses.html', slider_values=slider_values, textarea_content=textarea_content)
    return "No CSV file uploaded."


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
        return render_template('Personality_prediction/self-rating.html', opn_lemmatized_tokens=opn_lemmatized_tokens, csn_lemmatized_tokens=csn_lemmatized_tokens, ext_lemmatized_tokens=ext_lemmatized_tokens, agr_lemmatized_tokens=agr_lemmatized_tokens, neu_lemmatized_tokens=neu_lemmatized_tokens, opn_score=opn_score, csn_score=csn_score, ext_score=ext_score, agr_score=agr_score, neu_score=neu_score, opn_ended_score=opn_ended_score, textarea_content="", slider_values="")

    # Return the template for GET requests
    return render_template('Personality_prediction/self-rating.html')


@app.route('/predict_scores', methods=['POST', 'GET'])
def predict_scores():
    if request.method == 'POST':
        ext_1 = int(request.form['ext_1'])
        ext_2 = int(request.form['ext_2'])
        ext_3 = int(request.form['ext_3'])
        ext_4 = int(request.form['ext_4'])
        ext_5 = int(request.form['ext_5'])

        neu_1 = int(request.form['neu_1'])
        neu_2 = int(request.form['neu_2'])
        neu_3 = int(request.form['neu_3'])
        neu_4 = int(request.form['neu_4'])
        neu_5 = int(request.form['neu_5'])

        agr_1 = int(request.form['agr_1'])
        agr_2 = int(request.form['agr_2'])
        agr_3 = int(request.form['agr_3'])
        agr_4 = int(request.form['agr_4'])
        agr_5 = int(request.form['agr_5'])

        csn_1 = int(request.form['csn_1'])
        csn_2 = int(request.form['csn_2'])
        csn_3 = int(request.form['csn_3'])
        csn_4 = int(request.form['csn_4'])
        csn_5 = int(request.form['csn_5'])

        opn_1 = int(request.form['opn_1'])
        opn_2 = int(request.form['opn_2'])
        opn_3 = int(request.form['opn_3'])
        opn_4 = int(request.form['opn_4'])
        opn_5 = int(request.form['opn_5'])

        ext_score1 = (ext_1 + ext_2 + ext_3 + ext_4 + ext_5)/5
        neu_score1 = (neu_1 + neu_2 + neu_3 + neu_4 + neu_5)/5
        agr_score1 = (agr_1 + agr_2 + agr_3 + agr_4 + agr_5)/5
        csn_score1 = (csn_1 + csn_2 + csn_3 + csn_4 + csn_5)/5
        opn_score1 = (opn_1 + opn_2 + opn_3 + opn_4 + opn_5)/5

        opn_text = request.form['opn']
        csn_text = request.form['csn']
        ext_text = request.form['ext']
        agr_text = request.form['agr']
        neu_text = request.form['neu']

        # openness
        # Perform text processing (tokenization, lemmatization, etc.)
        opn_lemmatized_tokens = preprocess_text(opn_text)

        # Count occurrences of common words in lemmatized_tokens
        opn_word_count = sum(
            1 for word in opn_lemmatized_tokens if word in opn_target_words)

        opn_score2 = opn_word_count/len(opn_target_words)*5
        opn_score2 = round(opn_score2, 2)  # Round off to 2 decimal places

        # concientiousness
        # Perform text processing (tokenization, lemmatization, etc.)
        csn_lemmatized_tokens = preprocess_text(csn_text)

        # Count occurrences of common words in lemmatized_tokens
        csn_word_count = sum(
            1 for word in csn_lemmatized_tokens if word in csn_target_words)

        csn_score2 = csn_word_count/len(csn_target_words)*5
        csn_score2 = round(csn_score2, 2)  # Round off to 2 decimal places

        # extraversion
        # Perform text processing (tokenization, lemmatization, etc.)
        ext_lemmatized_tokens = preprocess_text(ext_text)

        # Count occurrences of common words in lemmatized_tokens
        ext_word_count = sum(
            1 for word in ext_lemmatized_tokens if word in ext_target_words)

        ext_score2 = ext_word_count/len(ext_target_words)*5
        ext_score2 = round(ext_score2, 2)  # Round off to 2 decimal places

        # agreeableness
        # Perform text processing (tokenization, lemmatization, etc.)
        agr_lemmatized_tokens = preprocess_text(agr_text)

        # Count occurrences of common words in lemmatized_tokens
        agr_word_count = sum(
            1 for word in agr_lemmatized_tokens if word in agr_target_words)

        agr_score2 = agr_word_count/len(agr_target_words)*5
        agr_score2 = round(agr_score2, 2)  # Round off to 2 decimal places

        # neuroticism
        # Perform text processing (tokenization, lemmatization, etc.)
        neu_lemmatized_tokens = preprocess_text(neu_text)

        # Count occurrences of common words in lemmatized_tokens
        neu_word_count = sum(
            1 for word in neu_lemmatized_tokens if word in neu_target_words)

        neu_score2 = neu_word_count/len(neu_target_words)*5
        neu_score2 = round(neu_score2, 2)  # Round off to 2 decimal places

        # Final Score
        opn_score = (opn_score1*0.4) + (opn_score2*0.6)
        opn_score = round(opn_score, 2)
        csn_score = (csn_score1*0.4) + (csn_score2*0.6)
        csn_score = round(csn_score, 2)
        ext_score = (ext_score1*0.4) + (ext_score2*0.6)
        ext_score = round(ext_score, 2)
        agr_score = (agr_score1*0.4) + (agr_score2*0.6)
        agr_score = round(agr_score, 2)
        neu_score = (neu_score1*0.4) + (neu_score2*0.6)
        neu_score = round(neu_score, 2)

        # expected scores
        # Retrieve the calculated values from the query parameters
        exp_openness = float(request.args.get('exp_openness'))
        exp_conscientiousness = float(
            request.args.get('exp_conscientiousness'))
        exp_extraversion = float(request.args.get('exp_extraversion'))
        exp_agreeableness = float(request.args.get('exp_agreeableness'))
        exp_neuroticism = float(request.args.get('exp_neuroticism'))

        # Pass the processed data to the template
        return render_template('Personality_prediction/results.html', ext_score1=ext_score1, neu_score1=neu_score1, agr_score1=agr_score1, csn_score1=csn_score1, opn_score1=opn_score1, ext_score2=ext_score2, neu_score2=neu_score2, agr_score2=agr_score2, csn_score2=csn_score2, opn_score2=opn_score2, ext_score=ext_score, neu_score=neu_score, agr_score=agr_score, csn_score=csn_score, opn_score=opn_score, exp_openness=exp_openness, exp_conscientiousness=exp_conscientiousness, textarea_content="", slider_values="")

    # Return the template for GET requests
    return render_template('Personality_prediction/results.html', ext_score1=ext_score1, neu_score1=neu_score1, agr_score1=agr_score1, csn_score1=csn_score1, opn_score1=opn_score1, ext_score2=ext_score2, neu_score2=neu_score2, agr_score2=agr_score2, csn_score2=csn_score2, opn_score2=opn_score2, ext_score=ext_score, neu_score=neu_score, agr_score=agr_score, csn_score=csn_score, opn_score=opn_score, exp_openness=exp_openness, exp_conscientiousness=exp_conscientiousness, textarea_content="", slider_values="")


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


@app.route('/calc_expected', methods=['POST', 'GET'])
def calcExpected():
    if request.method == 'POST':
        job_role = request.form.get('jobrole')
        innovative = float(request.form.get('innovative'))
        fast_learner = float(request.form.get('fastlearner'))
        organization_skills = float(request.form.get('organizationskills'))
        attention_to_detail = float(request.form.get('attentiontodetail'))
        assertiveness = float(request.form.get('assertiveness'))
        leadership_skills = float(request.form.get('leadershipskills'))
        team_player = float(request.form.get('teamplayer'))
        communication_skills = float(request.form.get('communicationskills'))
        confidence = float(request.form.get('confidence'))
        adaptability_to_changes = float(
            request.form.get('adaptabilitytochanges'))

        exp_openness = (fast_learner + innovative)/2
        exp_conscientiousness = (attention_to_detail + organization_skills)/2
        exp_extraversion = (assertiveness + leadership_skills)/2
        exp_agreeableness = (team_player + communication_skills)/2

        # expected (anti-)neuroticism score
        exp_neuroticism = (confidence + adaptability_to_changes)/2

    return render_template('Personality_prediction/requirement.html', job_role=job_role, exp_openness=exp_openness, exp_conscientiousness=exp_conscientiousness, exp_extraversion=exp_extraversion, exp_agreeableness=exp_agreeableness, exp_neuroticism=exp_neuroticism, textarea_content="", slider_values="")


if __name__ == '__main__':
    app.run(debug=True)
