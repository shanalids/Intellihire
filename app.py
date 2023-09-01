# personality prediction imports
from flask import Flask, request, url_for, redirect, render_template, render_template_string, session
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


# professional skills imports
from flask import Flask, render_template, request
from github import Github
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# cv analysis imports

# academic transcript imports




app = Flask(__name__)
app.secret_key = 'personality-prediction-2023'



@app.route('/')
def home():
    return render_template('home.html', textarea_content="", slider_values="")


# personality prediction - Maleesha - start
@app.route('/personality-home')
def personality_home():
    return render_template('personality_prediction/personality-home.html')

@app.route('/responses', methods=['GET', 'POST'])
def responses():
    return render_template('personality_prediction/responses.html', textarea_content="", slider_values="")


@app.route('/requirement', methods=['GET', 'POST'])
def requirement():
    return render_template('personality_prediction/requirement.html', textarea_content="", slider_values="")


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

            # Initialize variables to store textarea and slider values
            slider_values = []
            textarea_content = []

            # Read the last row of the CSV
            last_data_row = None
            for data_row in csv_reader:
                last_data_row = data_row

            if last_data_row:
                # Set the first value to the textarea
                textarea_first_value = last_data_row[0]

                # Extract values for sliders (next 25 values)
                for i in range(1, 26):
                    if i < len(last_data_row):
                        datum = last_data_row[i]
                        slider_values.append(int(datum))

                # Extract values for the last textareas (remaining 5 values)
                for i in range(26, 31):
                    if i < len(last_data_row):
                        datum = last_data_row[i]
                        textarea_content.append(datum)

                return render_template('personality_prediction/responses.html', textarea_first_value=textarea_first_value, slider_values=slider_values, textarea_content=textarea_content)

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


@app.route('/predict_scores', methods=['POST', 'GET'])
def predict_scores():
    if request.method == 'POST':
        # Retrieve calculated values from session
        exp_openness = session.get('exp_openness')
        exp_conscientiousness = session.get('exp_conscientiousness')
        exp_extraversion = session.get('exp_extraversion')
        exp_agreeableness = session.get('exp_agreeableness')
        exp_neuroticism = session.get('exp_neuroticism')

        name = (request.form['name'])

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

        # Matching Percentage
        opn = (opn_score/exp_openness)*20
        csn = (csn_score/exp_conscientiousness)*20
        ext = (ext_score/exp_extraversion)*20
        agr = (agr_score/exp_agreeableness)*20
        neu = (neu_score/exp_neuroticism)*20

        match_percentage = (opn+csn+ext+agr+neu)
        match_percentage = round(match_percentage, 2)

        # Pass the processed data to the template
        return render_template('personality_prediction/results.html', name=name, ext_score1=ext_score1, neu_score1=neu_score1, agr_score1=agr_score1, csn_score1=csn_score1, opn_score1=opn_score1, ext_score2=ext_score2, neu_score2=neu_score2, agr_score2=agr_score2, csn_score2=csn_score2, opn_score2=opn_score2, ext_score=ext_score, neu_score=neu_score, agr_score=agr_score, csn_score=csn_score, opn_score=opn_score, exp_openness=exp_openness, exp_conscientiousness=exp_conscientiousness, exp_extraversion=exp_extraversion, exp_agreeableness=exp_agreeableness, exp_neuroticism=exp_neuroticism, match_percentage=match_percentage, textarea_content="", slider_values="")

    # Return the template for GET requests
    return render_template('personality_prediction/results.html', name=name, ext_score1=ext_score1, neu_score1=neu_score1, agr_score1=agr_score1, csn_score1=csn_score1, opn_score1=opn_score1, ext_score2=ext_score2, neu_score2=neu_score2, agr_score2=agr_score2, csn_score2=csn_score2, opn_score2=opn_score2, ext_score=ext_score, neu_score=neu_score, agr_score=agr_score, csn_score=csn_score, opn_score=opn_score, exp_openness=exp_openness, exp_conscientiousness=exp_conscientiousness, exp_extraversion=exp_extraversion, exp_agreeableness=exp_agreeableness, exp_neuroticism=exp_neuroticism, match_percentage=match_percentage, textarea_content="", slider_values="")


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
        experience = request.form.get('experience')
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

        session['exp_openness'] = exp_openness
        session['exp_conscientiousness'] = exp_conscientiousness
        session['exp_extraversion'] = exp_extraversion
        session['exp_agreeableness'] = exp_agreeableness
        session['exp_openness'] = exp_openness
        session['exp_neuroticism'] = exp_neuroticism

    return render_template('personality_prediction/requirement.html', job_role=job_role, experience=experience, exp_openness=exp_openness, exp_conscientiousness=exp_conscientiousness, exp_extraversion=exp_extraversion, exp_agreeableness=exp_agreeableness, exp_neuroticism=exp_neuroticism, textarea_content="", slider_values="")

# personality prediction - Maleesha - END --------------------------------------------------------------------------------------------


# professional skills - Sandani - START ------------------------------------------------------------------------------------------------

# @app.route('/pf_home')
# def pf_home():
#     return render_template('professional_skills/pf_home.html')

# @app.route('/pf_home/plp_form')
# def plp_form():
#     return render_template('professional_skills/plp_form.html')

# @app.route('/pf_home/compare_form')
# def com_form():
#     return render_template('professional_skills/compare_form.html')

# @app.route('/pf_home/job_cat_form')
# def job_cat_form():
#     return render_template('professional_skills/job_cat_form.html')

def calculate_language_proficiency(username, access_token):
    # Authenticate with GitHub
    access_token = "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"
    g = Github(access_token)
    user = g.get_user(username)
    
    # Calculate language proficiency
    language_proficiency = {}
    repository_count = user.public_repos
    
    for repo in user.get_repos():
        languages = repo.get_languages()
        for language, value in languages.items():
            if language in language_proficiency:
                language_proficiency[language] += value
            else:
                language_proficiency[language] = value
    
    # Calculate weighted language proficiency scores
    weighted_scores = {}
    
    for language, value in language_proficiency.items():
        weighted_score = value * (1 / repository_count)
        weighted_scores[language] = weighted_score
    
    # Normalize and convert scores to percentages
    total_score = sum(weighted_scores.values())
    
    percentage_scores = {
        language: (score / total_score) * 100
        for language, score in weighted_scores.items()
    }
    
    return percentage_scores

def generate_pie_chart(percentage_scores):
    # Programming Language Proficiency Data
    languages = list(percentage_scores.keys())
    percentages = list(percentage_scores.values())
    
    # Create a pie chart
    fig, ax = plt.subplots()
    ax.pie(percentages, labels=languages, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)

    ax.axis('equal') # Equal aspect ratio ensures the pie chart is circular
    plt.title('Programming Language Proficiency (Percentage)')
    
    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    canvas.print_png(img)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         access_token = "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"
        
#         percentage_scores = calculate_language_proficiency(username, access_token)
#         pie_chart = generate_pie_chart(percentage_scores)
        
#         return render_template('index2.html', username=username, percentage_scores=percentage_scores, pie_chart=pie_chart)
    
#     return render_template('index2.html', username=None, percentage_scores=None, pie_chart=None)

@app.route('/')
def index():
    return render_template('professional_skills/plp_form.html')

@app.route('/plp', methods=['GET', 'POST'])
def plp():
    if request.method == 'POST':
        username = request.form.get('username')
        access_token = "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"
        
        percentage_scores = calculate_language_proficiency(username, access_token)
        pie_chart = generate_pie_chart(percentage_scores)
        
        return render_template('professional_skills/plp.html', username=username, percentage_scores=percentage_scores, pie_chart=pie_chart)
    
    return render_template('professional_skills/plp.html', username=None, percentage_scores=None, pie_chart=None)

# professional skills - Sandani - END-------------------------------------------------------------------------------------------------------


# CV Analysis - Manushi - START-------------------------------------------------------------------------------------------------------------
# CV Analysis - Manushi - END-----------------------------------------------------------------------------------------------------------------

# Academic Transcript - Shanali - START -------------------------------------------------------------------------------------------------------
#Academic Transcript - Shanali - END -----------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
