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
    return render_template('Personality_prediction/requirement.html', textarea_content="", slider_values="")

# @app.route('/')
# def index():
#     return render_template('Personality_prediction/open_ended.html', textarea_content="")


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

            return render_template('Personality_prediction/open_ended.html', textarea_content=textarea_content)
    return "No CSV file uploaded."


@app.route('/upload_selfrate', methods=['POST'])
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

            # Initialize a list to store slider values
            slider_values = []

            # Iterate over each data in the first row and convert them to integers
            for datum in data_row:
                # Convert the datum to an integer and append it to the slider values list
                slider_values.append(int(datum))

            return render_template('Personality_prediction/self-rating.html', slider_values=slider_values)
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


@app.route('/predict_selfrate', methods=['POST', 'GET'])
def predict_selfrate():
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

        ext_score = (ext_1 + ext_2 + ext_3 + ext_4 + ext_5)/5
        neu_score = (neu_1 + neu_2 + neu_3 + neu_4 + neu_5)/5
        agr_score = (agr_1 + agr_2 + agr_3 + agr_4 + agr_5)/5
        csn_score = (csn_1 + csn_2 + csn_3 + csn_4 + csn_5)/5
        opn_score = (opn_1 + opn_2 + opn_3 + opn_4 + opn_5)/5

        # Pass the processed data to the template
        return render_template('Personality_prediction/self-rating.html', ext_score=ext_score, neu_score=neu_score, agr_score=agr_score, csn_score=csn_score, opn_score=opn_score, textarea_content="", slider_values="")

    # Return the template for GET requests
    return render_template('Personality_prediction/self-rating.html')


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
        innovative = int(request.form.get('innovative'))
        fast_learner = int(request.form.get('fastlearner'))
        organization_skills = int(request.form.get('organizationskills'))
        attention_to_detail = int(request.form.get('attentiontodetail'))
        assertiveness = int(request.form.get('assertiveness'))
        leadership_skills = int(request.form.get('leadershipskills'))
        team_player = int(request.form.get('teamplayer'))
        communication_skills = int(request.form.get('communicationskills'))
        confidence = int(request.form.get('confidence'))
        adaptability_to_changes = int(
            request.form.get('adaptabilitytochanges'))

        exp_openness = (fast_learner + innovative)/2
        exp_conscientiousness = (attention_to_detail + organization_skills)/2
        exp_extraversion = (assertiveness + leadership_skills)/2
        exp_agreeableness = (team_player + communication_skills)/2

        # expected (anti-)neuroticism score
        exp_neuroticism = (confidence + adaptability_to_changes)/2

    return render_template('Personality_prediction/requirement.html', job_role=job_role, exp_openness=exp_openness, exp_conscientiousness=exp_conscientiousness, exp_extraversion=exp_extraversion, exp_agreeableness=exp_agreeableness, exp_neuroticism=exp_neuroticism, textarea_content="", slider_values="")


# @app.route('/bar_chart', methods=['POST', 'GET'])
# def bar_chart():
#     if request.method == 'POST':
#         # Get the scores from the form
#         opn_score = float(request.form['opn_score'])
#         csn_score = float(request.form['csn_score'])
#         ext_score = float(request.form['ext_score'])
#         agr_score = float(request.form['agr_score'])
#         neu_score = float(request.form['neu_score'])
#         opn_ended_score = float(request.form['opn_ended_score'])

#         # Labels for the personality traits
#         traits = ['Openness', 'Conscientiousness', 'Extraversion',
#                   'Agreeableness', 'Neuroticism', 'Final Score']

#         # Corresponding scores
#         scores = [opn_score, csn_score, ext_score,
#                   agr_score, neu_score, opn_ended_score]

#         # Create a bar chart
#         plt.figure(figsize=(10, 6))
#         plt.bar(traits, scores, color=[
#                 'blue', 'green', 'red', 'purple', 'orange', 'gray'])
#         plt.xlabel('Personality Traits')
#         plt.ylabel('Scores')
#         plt.title('Personality Trait Scores')
#         plt.ylim(0, 5)  # Set y-axis range
#         plt.tight_layout()

#         # Save the bar chart as a PNG image
#         chart_path = 'static/personality_scores.png'
#         plt.savefig(chart_path)

#         # Pass the chart path to the template
#         return render_template('Personality_prediction/bar_chart.html', chart_path=chart_path)

#     # Return the template for GET requests
#     return render_template('Personality_prediction/open_ended.html')


if __name__ == '__main__':
    app.run(debug=True)
