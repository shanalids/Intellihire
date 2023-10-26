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
# nltk.download('punkt')
# nltk.download('stopwords')


# professional skills imports
from flask import Flask, render_template, request
from github import Github
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from flask import Flask, render_template, request
import pickle

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import io
import base64

# cv analysis imports
from flask import Flask, render_template, request, session
import joblib
import json
import re
from tabulate import tabulate
import pandas as pd
import nltk
from backend.cv_analysis.process import *
import numpy as np
import pickle
import backend.cv_analysis.file_new as dp
from backend.cv_analysis.helper import *
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from flask import jsonify


# academic transcript imports
from flask import Flask, render_template, request, send_file
import os
import spacy
import glob
import re
import pytesseract
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



app = Flask(__name__)
app.secret_key = 'personality-prediction-2023'


# @app.route('/')
# def home():
#     return render_template('home.html', textarea_content="", slider_values="")

@app.route('/')
def home():
    return render_template('new_home.html', textarea_content="", slider_values="")



# personality prediction - Maleesha - start
@app.route('/requirement')
def personality_home():
    return render_template('personality_prediction/requirement.html')

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
        session['cand_name'] = name

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
        if(opn_score<exp_openness):
            opn = (opn_score/exp_openness)
        else:
            opn = 0.9

        if(csn_score<exp_conscientiousness):
            csn = (csn_score/exp_conscientiousness)
        else:
            csn = 0.9

        if(ext_score<exp_extraversion):
            ext = (ext_score/exp_extraversion)
        else:
            ext = 0.9

        if(agr_score<exp_agreeableness):
            agr = (agr_score/exp_agreeableness)
        else:
            agr = 0.9

        if(neu_score<exp_neuroticism):
            neu = (neu_score/exp_neuroticism)
        else:
            neu = 0.9

        match_percentage = (opn+csn+ext+agr+neu)*20
        match_percentage = round(match_percentage, 2)

        # Load the K-Means model from the pickle file
        with open('models/personality_prediction/kmeans_model.pkl', 'rb') as model_file:
            k_means_model = pickle.load(model_file)

        # Prepare the input data as a list or array
        input_data = [[ext_1, ext_2, ext_3, ext_4, ext_5, neu_1, neu_2, neu_3, neu_4, neu_5, agr_1, agr_2, agr_3, agr_4, agr_5, csn_1, csn_2, csn_3, csn_4, csn_5, opn_1, opn_2, opn_3, opn_4, opn_5]]

        # Use the loaded model to make predictions
        cluster_prediction = k_means_model.predict(input_data)

        session['personality_score'] = match_percentage

        # Pass the processed data to the template
        return render_template('personality_prediction/results.html', name=name, ext_score1=ext_score1, neu_score1=neu_score1, agr_score1=agr_score1, csn_score1=csn_score1, opn_score1=opn_score1, ext_score2=ext_score2, neu_score2=neu_score2, agr_score2=agr_score2, csn_score2=csn_score2, opn_score2=opn_score2, ext_score=ext_score, neu_score=neu_score, agr_score=agr_score, csn_score=csn_score, opn_score=opn_score, exp_openness=exp_openness, exp_conscientiousness=exp_conscientiousness, exp_extraversion=exp_extraversion, exp_agreeableness=exp_agreeableness, exp_neuroticism=exp_neuroticism, match_percentage=match_percentage, cluster_prediction=cluster_prediction, textarea_content="", slider_values="")

    # Return the template for GET requests
    return render_template('personality_prediction/results.html', name=name, ext_score1=ext_score1, neu_score1=neu_score1, agr_score1=agr_score1, csn_score1=csn_score1, opn_score1=opn_score1, ext_score2=ext_score2, neu_score2=neu_score2, agr_score2=agr_score2, csn_score2=csn_score2, opn_score2=opn_score2, ext_score=ext_score, neu_score=neu_score, agr_score=agr_score, csn_score=csn_score, opn_score=opn_score, exp_openness=exp_openness, exp_conscientiousness=exp_conscientiousness, exp_extraversion=exp_extraversion, exp_agreeableness=exp_agreeableness, exp_neuroticism=exp_neuroticism, match_percentage=match_percentage, cluster_prediction=cluster_prediction, textarea_content="", slider_values="")


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

        session['jobrole'] = job_role
        session['exp_openness'] = exp_openness
        session['exp_conscientiousness'] = exp_conscientiousness
        session['exp_extraversion'] = exp_extraversion
        session['exp_agreeableness'] = exp_agreeableness
        session['exp_openness'] = exp_openness
        session['exp_neuroticism'] = exp_neuroticism

    return render_template('personality_prediction/requirement.html', job_role=job_role, experience=experience, exp_openness=exp_openness, exp_conscientiousness=exp_conscientiousness, exp_extraversion=exp_extraversion, exp_agreeableness=exp_agreeableness, exp_neuroticism=exp_neuroticism, textarea_content="", slider_values="")

# personality prediction - Maleesha - END --------------------------------------------------------------------------------------------


# professional skills - Sandani - START ------------------------------------------------------------------------------------------------

@app.route('/pf_home')
def pf_home():
    return render_template('professional_skills/pf_home.html')

# @app.route('/pf_home/plp_form')
# def plp_form():
#     return render_template('plp_form.html')

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


@app.route('/pf_home/plp_form')
def plp_form():
    return render_template('professional_skills/plp_form.html')

@app.route('/pf_home/plp_form/plp', methods=['GET', 'POST'])
def plp():
    if request.method == 'POST':
        username = request.form.get('username')
        access_token = "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"
        
        percentage_scores = calculate_language_proficiency(username, access_token)
        session['percentage_scores'] = percentage_scores

        pie_chart = generate_pie_chart(percentage_scores)
        
        return render_template('professional_skills/plp.html', username=username, percentage_scores=percentage_scores, pie_chart=pie_chart)
    
    return render_template('professional_skills/plp.html', username=None, percentage_scores=None, pie_chart=None)

#----------------git-compare---------------------------
def get_language_proficiency(username):
    # Authenticate with GitHub
    access_token = "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"
    g = Github(access_token)
    user = g.get_user(username)
    
    language_proficiency = {}
    repository_count = user.public_repos
    
    for repo in user.get_repos():
        languages = repo.get_languages()
        for language, value in languages.items():
            if language in language_proficiency:
                language_proficiency[language] += value
            else:
                language_proficiency[language] = value
                
    weighted_scores = {}
    for language, value in language_proficiency.items():
        weighted_score = value * (1 / repository_count)
        weighted_scores[language] = weighted_score
        
    total_score = sum(weighted_scores.values())
    percentage_scores = {
        language: (score / total_score) * 100
        for language, score in weighted_scores.items()
    }
    
    return percentage_scores

def generate_comparison_chart(current_user_scores, other_user_scores, common_languages):
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.4
    index = range(len(common_languages))

    bar1 = ax.bar(index, current_user_scores, bar_width, label='Current candidate', color='b', alpha=0.7)
    bar2 = ax.bar([i + bar_width for i in index], other_user_scores, bar_width, label='Peer candidate', color='g', alpha=0.7)

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=10)

    add_labels(bar1)
    add_labels(bar2)

    ax.set_xlabel('Programming Languages')
    ax.set_ylabel('Proficiency (%)')
    ax.set_title('Common Language Proficiency Comparison')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(common_languages, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    img_base64 = base64.b64encode(img_stream.read()).decode('utf-8')
    plt.close()

    return img_base64

@app.route('/pf_home/compare_form')
def compare_form():
    return render_template('professional_skills/compare_form.html')

@app.route('/pf_home/compare_form/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'POST':
        current_username = request.form['current_username']
        other_username = request.form['other_username']

        current_user_scores = get_language_proficiency(current_username)
        other_user_scores = get_language_proficiency(other_username)

        # Compare language proficiency
        common_languages = list(set(current_user_scores.keys()) & set(other_user_scores.keys()))

        img_base64 = generate_comparison_chart(
            [current_user_scores.get(lang, 0) for lang in common_languages],
            [other_user_scores.get(lang, 0) for lang in common_languages],
            common_languages
        )

        return render_template('professional_skills/compare.html', img_base64=img_base64)

    return render_template('professional_skills/compare.html')

#------------LinkedIn job category ---------------------------------------------------------------------------------------------

def scrape_linkedin_skills(linkedin_profile_url):
    api_key = 'YMJ1T-ETnLHCCtrAvSE1Xg'
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    headers = {'Authorization': 'Bearer ' + api_key}

    response = requests.get(api_endpoint,
                            params={'url': linkedin_profile_url, 'skills': 'include'},
                            headers=headers)

    profile_data = response.json()

    if 'skills' in profile_data:
        return profile_data['skills']
    else:
        return []

link_model = pickle.load(open("models/professional_skills/model.pkl", "rb"))
fitted_vectorizer = pickle.load(open("models/professional_skills/fitted_vectorizer.pkl", "rb"))

@app.route('/pf_home/job_cat_form', methods=['GET', 'POST'])
def job_cat():
    result = None
    if request.method == 'POST':
        linkedin_profile_url = request.form['linkedin_profile_url']
        skills = scrape_linkedin_skills(linkedin_profile_url)  # Calling scraping function
        if skills:
            #predicted_category = model.predict([skills])  #model takes a list of skills
            predicted_category = link_model.predict(fitted_vectorizer.transform(skills))

            session['predicted_category'] = predicted_category

            result = f"Predicted Job Category: {predicted_category[0]}"
    return render_template('professional_skills/job_cat_form.html', result=result)

#------sentiment analysis---------------
# Function to perform sentiment analysis and visualization for a specific row
def analyze_sentiment_and_visualize(row):
    # Step 2: Perform sentiment analysis
    sid = SentimentIntensityAnalyzer()

    # Calculate sentiment scores for each column
    sentiment_scores = {}
    for column, response in row.items():
        response = str(response)
        score = sid.polarity_scores(response)
        sentiment_scores[column] = score

    # Calculate overall sentiment distribution
    labels = ['Positive', 'Negative', 'Neutral']
    sentiment_distribution = [0, 0, 0]

    for score in sentiment_scores.values():
        max_sentiment = max(score, key=score.get)
        
        if max_sentiment == 'pos':
            sentiment_distribution[0] += 1
        elif max_sentiment == 'neg':
            sentiment_distribution[1] += 1
        else:
            sentiment_distribution[2] += 1

    # Calculate positive percentage
    positive_percentage = (sentiment_distribution[0] / sum(sentiment_distribution)) * 100

    # Store positive percentage in the session
    session['positive_percentage'] = positive_percentage

    # Create a single pie chart for sentiment distribution
    colors = ['blue', 'red', 'green']
    plt.figure()
    plt.pie(sentiment_distribution, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Analysis')
    plt.axis('equal')

    # Save the plot to a BytesIO object
    sentiment_plot = io.BytesIO()
    plt.savefig(sentiment_plot, format='png')
    sentiment_plot.seek(0)
    plt.close()

    # Convert the plot image to a base64-encoded string
    plot_data = base64.b64encode(sentiment_plot.getvalue()).decode('utf-8')
    return plot_data

@app.route('/pf_home/sentiment')
def senti1():
    return render_template('professional_skills/sentiment.html')


@app.route('/pf_home/sentiment/sentiment_results', methods=['GET', 'POST'])
def senti2():
    plot_data = None
    column_names = None
    row_content = None
    
    if request.method == 'POST':
        # Get uploaded file
        uploaded_file = request.files['file']
        
        if uploaded_file.filename != '':
            data = pd.read_csv(uploaded_file)
            columns_to_analyze = data.columns[1:]
            data = data[columns_to_analyze]

            last_row = data.iloc[-1]

            text = last_row.iloc[3]
            tokenized_strengths = extract_strengths(text)

            # session['tokenized_strengths'] = tokenized_strengths
            
            # Update sentiment analysis and visualization function to return column names
            column_names = last_row.index.tolist()
            plot_data = analyze_sentiment_and_visualize(last_row)
            row_content = last_row.values.tolist()

    return render_template('professional_skills/sentiment_results.html', plot_data=plot_data, column_names=column_names, row_content=row_content)

def extract_strengths(text):
    # Tokenize the text using NLTK
    tokens = word_tokenize(text)
    
    # Calculate the frequency of each token
    token_freq = Counter(tokens)
    
    # Sort tokens by frequency in descending order
    sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Extract all unique tokens as keywords
    keywords = [token for token, freq in sorted_tokens]
    
    return keywords


@app.route('/pf_home/sentiment/sentiment_results', methods=['GET', 'POST'])
def senti3():
    plot_data = None
    if request.method == 'POST':
        # Get uploaded file
        uploaded_file = request.files['file']
        
        # Check if a file was uploaded
        if uploaded_file.filename != '':
            # Read uploaded CSV file into a DataFrame
            data = pd.read_csv(uploaded_file)
            
            # Exclude the first column (timestamp)
            columns_to_analyze = data.columns[1:]
            data = data[columns_to_analyze]

            # Get the last row in the DataFrame
            last_row = data.iloc[-1]

            # Apply sentiment analysis and visualization to the last row
            plot_data = analyze_sentiment_and_visualize(last_row)

    return render_template('professional_skills/sentiment_results.html', plot_data=plot_data)


# professional skills - Sandani - END-------------------------------------------------------------------------------------------------------


# CV Analysis - Manushi - START-------------------------------------------------------------------------------------------------------------

#load models for cv_analysis
model = pickle.load(open("models/cv_analysis/stkmodel.pkl", "rb"))
saved_filename = "models/cv_analysis/Vectorizer1.joblib"
vectorizer = joblib.load(saved_filename)


@app.route('/ranking', methods=['GET', 'POST'])
def ranking():
     # Define session_data as an empty dictionary
    session_data = {} 


    # Getting uploaded files
    if request.method == 'POST':
        uploaded_pdfs = request.files.getlist('pdf')
        uploaded_jd = request.files.get('jd')
        skills_list = request.form.get('skills')  # Get the entered skills from the form
        
        jd_text = " "
        jd = " "

        # Start preprocessing with jd

        if uploaded_jd and uploaded_jd.filename.endswith('.pdf'):

            jd = extract_pdf_text(uploaded_jd)
            jd_text = preprocess_text_resume(jd)
            
            keywords_jd = dp.spacy_keywords(jd_text)
            

            
            
            matching_results = []

            for uploaded_pdf in uploaded_pdfs:
                if uploaded_pdf.filename.endswith('.pdf'):

                    # Preprocessing of resume
                    pdf = extract_pdf_text(uploaded_pdf)
                    pdf_text = preprocess_text_resume(pdf)
                    
                    
                    # Keyword extraction of resume
                    keywords_resume = dp.nltk_keywords(pdf_text)

                    # Start matching prediction
                    # Join the elements of the list into a single string with space as a separator
                    #concatenated_text = ' '.join(pdf_text)  # Assuming pdf_text is a list of strings
                    #jd_text = str(jd_text) 
                    # Now you can concatenate the resulting string with jd_text
                    #new_data_transformed = vectorizer.transform([concatenated_text + " " + jd_text])
                    new_data_transformed = vectorizer.transform([pdf_text + " " + jd_text])
                    new_data_prediction = model.predict(new_data_transformed)
                    
                    new_data_prediction = model.predict(new_data_transformed)

                    # Round matching percentage to nearest 2 decimals
                    new_data_prediction = np.round(new_data_prediction, 2)

                    jd_keywords_in_resume_table = []
                    matching_keywords = []

                    # Iterate loop for each keyword in resume

                    for word in keywords_jd:
                        if word in keywords_resume:
                            match_result = [word, 'Match']
                            matching_keywords.append(word)
                        else:
                            match_result = [word, 'No Match']
                        jd_keywords_in_resume_table.append(match_result)

                    
                    extracted_skills_frequency = extract_skills_and_count_frequency(pdf, skills_list)

                    skills=extract_skills(pdf,skills_list)

                  

                    # Assuming profile.extracted_skills_frequency is a dictionary
                    skills_data_json = json.dumps(extracted_skills_frequency)


                    # Store the analysis results in the session_data dictionary
                    session_data['results_' + uploaded_pdf.filename] = {
                        "pdf_filename": uploaded_pdf.filename,
                        "matching_percentage": new_data_prediction[0],
                        "skills":skills,
                        "jd_keywords_in_resume_table": jd_keywords_in_resume_table,
                        "matching_keywords": matching_keywords,
                        "extracted_skills_frequency": extracted_skills_frequency,
                        "skills_data_json":skills_data_json
                    }
                    
                    # Append matching results
                    matching_results.append({
                        "pdf_filename": uploaded_pdf.filename,
                        "matching_percentage": new_data_prediction[0],
                        "skills":skills,
                        "jd_keywords_in_resume_table": jd_keywords_in_resume_table,
                        "matching_keywords": matching_keywords,

                        "extracted_skills_frequency": extracted_skills_frequency,
                        "skills_data_json":skills_data_json
                    })

            # Store matching_results in the session
            session['matching_results'] = matching_results

            # Rank matching results and render the template
            
            ranking = sorted(matching_results, key=lambda x: x["matching_percentage"], reverse=True)

            session['ranking'] = ranking
            return redirect(url_for('results'))
            # return render_template('cv_analysis/results.html', ranking=ranking)

    return render_template('cv_analysis/index.html', ranking=[])

# Route for displaying results
@app.route('/results')
def results():
    # Retrieve the matching results and ranking from the session
    # matching_results = session.get('matching_results', [])
    ranking = session.get('ranking', [])

    # You can perform additional processing or formatting of data if needed

    # return render_template('cv_analysis/results.html', ranking=ranking, matching_results=matching_results)
    return render_template('cv_analysis/results.html', ranking=ranking)

@app.route('/profile/<int:pdf_index>')
def view_profile(pdf_index):
    if 'matching_results' in session:
        matching_results = session['ranking']
        
        
        # Check if the selected PDF index is within the valid range
        if pdf_index >= 0 and pdf_index < len(matching_results):
            selected_profile = matching_results[pdf_index]
            session['selected_matching_percentage'] = selected_profile['matching_percentage']
            return render_template('CV_analysis/profile.html', profile=selected_profile)
            
    # Handle the case when the PDF index is invalid or matching results are not available
    return "Profile not found", 404
    

def extract_skills(resume_text, skills_list):
    # Create a regex pattern to match skills (case-insensitive)
    skills_pattern = r'\b(?:' + '|'.join(re.escape(skill.strip()) for skill in skills_list.split(',')) + r')\b'
    
    # Find all matches of skills in the resume text
    matches = re.findall(skills_pattern, resume_text, flags=re.IGNORECASE)
    
    return matches



def extract_skills_and_count_frequency(resume_text, skills_list):
    # Create a regex pattern to match skills (case-insensitive)
    skills_pattern = r'\b(?:' + '|'.join(re.escape(skill.strip()) for skill in skills_list.split(',')) + r')\b'
    
    # Find all matches of skills in the resume text
    matches = re.findall(skills_pattern, resume_text, flags=re.IGNORECASE)
    
    # Count the frequency of each skill
    skill_frequency = Counter(matches)
    
    return skill_frequency


#------------chatbot -START-----------------------------------------------



@app.route('/bot')
def chatbot():
    
    return render_template('CV_analysis/chatbot-home.html')


@app.route('/get_bot_response', methods=['POST'])
def get_bot_response():
    user_message = request.json['message']

    rasa_response = requests.post('http://localhost:5005/webhooks/rest/webhook', json={'message': user_message})
    bot_response = rasa_response.json()[0]['text']

    return jsonify({'bot_response': bot_response})


@app.route('/download_pdf')
def download_pdf():

    pdf_filename = "chatbot/output.pdf"
    return send_file(pdf_filename, as_attachment=True)


#------------chatbot -END------------------------------------------------



# CV Analysis - Manushi - END-----------------------------------------------------------------------------------------------------------------


# Academic Transcript - Shanali - START -------------------------------------------------------------------------------------------------------

# Load the trained NER model from disk
nlp_ner = spacy.load('models/academic_transcript/IdentifyingModulesandGrades')

# Load the module outline CSV file
module_outline_df = pd.read_csv("files/academic_transcript/Module Keywords.csv")

 

# Clean and preprocess text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = text.strip()

    return text

 

@app.route("/AcademicTranscript")

def index():
    return render_template('academic_transcript/AcaedmicTranscriptsIndex.html')

 

@app.route('/extract', methods=['POST'])

def extract_text():

    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        # Save the uploaded file temporarily
        uploaded_file.save('files/academic_transcript/uploaded_file.pdf')

        pdf_paths = glob.glob("files/academic_transcript/uploaded_file.pdf")
        text_file_path = "files/academic_transcript/extracted_text.txt"

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Provide the path to your Tesseract executable

        with open(text_file_path, "w", encoding="utf-8") as text_file:
            for pdf_path in pdf_paths:
                pages = convert_from_path(pdf_path, dpi=300)  # Adjust DPI as needed

                for pageNum, imBlob in enumerate(pages):
                    im = Image.frombytes('RGB', imBlob.size, imBlob.tobytes())  # Corrected line
                    im = im.convert('L')  # Convert to grayscale
                    im = im.point(lambda x: 0 if x < 200 else x)  # Threshold to remove light noise
                    text = pytesseract.image_to_string(im, lang='eng', config='--psm 6')  # Adjust configuration
                    text_file.write(text)

        # Read the extracted text from the file with proper encoding
        with open('files/academic_transcript/extracted_text.txt', 'r', encoding='utf-8') as file:
            extracted_text = file.read()
        #print(extracted_text)

        # Extract sections using regular expressions
        pattern = r'YEAR (\d+)\n([\s\S]+?)\nYear \1 Credits='
        matches = re.findall(pattern, extracted_text)
        sections = [match[1] for match in matches]
        #print(sections)

        # Save sections to a text file
        output_text_file_path = "files/academic_transcript/extracted_sections.txt"
        with open(output_text_file_path, "w") as output_text_file:
            for section in sections:
                output_text_file.write(section + "\n\n")

        # Load the document from the text file
        with open("files/academic_transcript/extracted_sections.txt", "r") as file:
            text = file.read()

        doc = nlp_ner(text)
        for ent in doc.ents:
            print(ent.text, ent.label_)

        def test_ner_model(file_path, output_file):
            with open(file_path, 'r') as file:
                 lines = file.readlines()

            # Process sections with custom NER model
            module_titles = []
            grades = []

            for line in lines:
                doc = nlp_ner(line.strip())
                module_title = None
                grade = None

                for ent in doc.ents:
                    if ent.label_ == "MODULE_TITLE":
                        module_title = ent.text
                    elif ent.label_ == "GRADE":
                        grade = ent.text

                if module_title and grade:  # Only append if both module title and grade exist
                    module_titles.append(module_title)
                    grades.append(grade)
            # print(extracted_data)


        # Create a DataFrame with extracted data
            data = {"Module Title": module_titles, "Grade": grades}
            df = pd.DataFrame(data)

            # Save the DataFrame to a CSV file
            df.to_csv(output_file, index=False)
            print("Saved extracted data to", output_file)

       

        # Provide the file path to test and output file path
        file_path = "files/academic_transcript/extracted_sections.txt"
        output_file = "files/academic_transcript/extracted_data.csv"
        test_ner_model(file_path, output_file)

        # Initialize CountVectorizer
        countvect = CountVectorizer()


        model = 'models/academic_transcript/skill_prediction.pkl'

        # Load the model from the file
        with open( model, 'rb') as model_file:
            classification_model = pickle.load(model_file)


        # Split the tuple to access the model and vectorizer
        model, countvect = classification_model

        #print(classification_model)
        # Assuming your classification_model tuple is defined as follows:
        # classification_model = (SVC(kernel='linear', random_state=0), CountVectorizer())

        #print(model)

        # Fit the model on training data (you should do this before calling predict)

        # model.fit(X_train, y_train)

        # Predict using the fitted model
        #predicted_categories = model.predict(module_title_vectors)

        # Assuming your classification_model tuple is defined as follows:
        # classification_model = (SVC(kernel='linear', random_state=0), CountVectorizer())

 

        # Split the tuple to access the model and vectorizer
        model, countvect = classification_model

        # Read module titles from the first CSV file
        module_titles_df = pd.read_csv('files/academic_transcript/extracted_data.csv')

        # Extract module titles from the DataFrame
        module_titles = module_titles_df['Module Title']

        # Read module titles and module keywords from the second CSV file
        module_keywords_df = pd.read_csv('files/academic_transcript/Module Keywords.csv')

        # Create a dictionary to map module titles to their corresponding module keywords
        module_title_to_keywords = dict(zip(module_keywords_df['Module Title'], module_keywords_df['Module Keywords']))

 
        # Define a function to predict skill areas
        def predict_skill_area(module_titles, module_title_to_keywords):
            predicted_categories = []
            for module_title in module_titles:
                module_keywords = module_title_to_keywords.get(module_title, "")
                combined_text = module_title.lower() + ' ' + module_keywords.lower()
                module_title_vector = countvect.transform([combined_text])
                predicted_category = model.predict(module_title_vector)
                predicted_categories.append(predicted_category[0])  # Assuming you want a single prediction
            return predicted_categories


        # Predict skill areas using the function
        predicted_skill_areas = predict_skill_area(module_titles, module_title_to_keywords)

        # Map the module titles from the extracted data to the module titles in the module outline
        module_titles_df['Category'] = predicted_skill_areas

        # Define the grade weighting dictionary
        grade_weighting = {
            'A+': 10,
            'A': 9,
            'A-': 8,
            'B+': 7,
            'B': 6,
            'B-': 5,
            'C+': 4,
            'C': 3,
            'C-': 2,
            'D+': 1,
            'D': 1,
            'E': 1
        }

        # Calculate the weighted grades based on the grade weighting dictionary
        module_titles_df['Weighted Grade'] = module_titles_df['Grade'].map(grade_weighting)

        # Create a table with the skill areas, module titles, module descriptions, and weighted grades
        skill_area_table = module_titles_df[['Category', 'Module Title', 'Weighted Grade']]

        unique_categories = skill_area_table['Category'].unique()
        print(unique_categories)

        unique_categories = skill_area_table['Category'].unique()

        # Define the predefined categories
        predefined_categories = [
            'System Administration',
            'IT Infrastructure and Networking',
            'Data Science and Analytics',
            'Artificial Intelligence and Machine Learning',
            'Cloud Computing',
            'Cybersecurity',
            'Database Management',
            'Project Management',
            'Programming and Software Development',
            'User Experience and Design'
        ]

        # Find categories not present in unique_categories
        missing_categories = [category for category in predefined_categories if category not in unique_categories]

        # Print the missing categories
        print("Missing Categories:", missing_categories)

        # Create a DataFrame for missing categories and assign 0.0 as 'Weighted Grade'
        missing_categories_df = pd.DataFrame({'Category': missing_categories})
        missing_categories_df['Weighted Grade'] = 0.0

        # Print the DataFrame with missing categories and their assigned weighted grades
        print("Missing Categories DataFrame:")
        print(missing_categories_df)

        #Exclude module titles with category "Core Competencies"
        skill_area_table = skill_area_table[skill_area_table['Category'] != "core competencies and soft skills"]


        # Calculate the weighted grades based on the grade weighting dictionary
        module_titles_df['Weighted Grade'] = module_titles_df['Grade'].map(grade_weighting)
        
        # Group the skill area table by the 'Category' column and calculate the sum of the 'Weighted Grade' column for each category
        category_totals = module_titles_df.groupby('Category')['Weighted Grade'].sum()


        # Sort the categories by their total weighted grades in descending order
        category_totals = category_totals.sort_values(ascending=False)

        # print("cat tot:",category_totals)


        # Merge missing_categories_df and category_totals into one DataFrame
        combined_category_totals = pd.concat([missing_categories_df, category_totals.reset_index()], ignore_index=True)

        # # Sort the combined DataFrame by 'Weighted Grade' in descending order
        # combined_category_totals = combined_category_totals.sort_values(by='Weighted Grade', ascending=False)

        # Print the combined DataFrame
        print("Combined Category Totals:")
        print(combined_category_totals)

        # Add a dictionary to map job titles to skill weights
        job_title_to_skill_weights = {
            'Software Engineer': {
                'Programming and Software Development': 10,
                'Data Science and Analytics': 5,
                'Database Management': 4,
                'Cloud Computing': 6,
                'Project Management': 3,
                'Cybersecurity': 4,
                'IT Infrastructure and Networking': 5,
                'Artificial Intelligence and Machine Learning': 7,
                'System Administration': 3,
                'User Experience and Design': 4,
            },
            'Data Scientist': {
                'Programming and Software Development': 5,
                'Data Science and Analytics': 10,
                'Database Management': 5,
                'Cloud Computing': 4,
                'Project Management': 3,
                'Cybersecurity': 3,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 9,
                'System Administration': 2,
                'User Experience and Design': 3,
            },

             'Data Engineer': {
                'Programming and Software Development': 7,
                'Data Science and Analytics': 10,
                'Database Management': 9,
                'Cloud Computing': 8,
                'Project Management': 3,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 3,
                'Artificial Intelligence and Machine Learning': 9,
                'System Administration': 2,
                'User Experience and Design': 4,
            },


            'Machine Learning Engineer': {
                'Programming and Software Development': 6,
                'Data Science and Analytics': 8,
                'Database Management': 3,
                'Cloud Computing': 4,
                'Project Management': 3,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 3,
                'Artificial Intelligence and Machine Learning': 10,
                'System Administration': 2,
                'User Experience and Design': 4,
            },

            'Data Analyst': {
                'Programming and Software Development': 3,
                'Data Science and Analytics': 9,
                'Database Management': 4,
                'Cloud Computing': 2,
                'Project Management': 3,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 3,
                'System Administration': 2,
                'User Experience and Design': 2,
            },

            'Database Administrator': {
                'Programming and Software Development': 3,
                'Data Science and Analytics': 4,
                'Database Management': 9,
                'Cloud Computing': 2,
                'Project Management': 3,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 6,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 7,
                'User Experience and Design': 2,
            },
            
            'Project Manager': {
                'Programming and Software Development': 7,
                'Data Science and Analytics': 2,
                'Database Management': 3,
                'Cloud Computing': 4,
                'Project Management': 10,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 4,
                'Artificial Intelligence and Machine Learning': 3,
                'System Administration': 2,
                'User Experience and Design': 3,
            },

            'Cyber Security Analyst': {
                'Programming and Software Development': 6,
                'Data Science and Analytics': 7,
                'Database Management': 2,
                'Cloud Computing': 3,
                'Project Management': 2,
                'Cybersecurity': 10,
                'IT Infrastructure and Networking': 5,
                'Artificial Intelligence and Machine Learning': 5,
                'System Administration': 4,
                'User Experience and Design': 2,
            },

            'Network Engineer': {
                'Programming and Software Development': 2,
                'Data Science and Analytics': 2,
                'Database Management': 4,
                'Cloud Computing': 3,
                'Project Management': 2,
                'Cybersecurity': 5,
                'IT Infrastructure and Networking': 9,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 6,
                'User Experience and Design': 2,
            },

            
            'UI/UX Designer': {
                'Programming and Software Development': 2,
                'Data Science and Analytics': 2,
                'Database Management': 2,
                'Cloud Computing': 2,
                'Project Management': 3,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 4,
                'System Administration': 2,
                'User Experience and Design': 10,
            },

            'Cyber Security Engineer': {
                'Programming and Software Development':7,
                'Data Science and Analytics': 2,
                'Database Management': 3,
                'Cloud Computing': 3,
                'Project Management': 2,
                'Cybersecurity': 10,
                'IT Infrastructure and Networking': 7,
                'Artificial Intelligence and Machine Learning': 5,
                'System Administration': 4,
                'User Experience and Design': 2,
            },

            'AI Engineer': {
                'Programming and Software Development': 7,
                'Data Science and Analytics': 8,
                'Database Management': 3,
                'Cloud Computing': 4,
                'Project Management': 3,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 3,
                'Artificial Intelligence and Machine Learning': 10,
                'System Administration': 2,
                'User Experience and Design': 4,
            },

            
            'Cloud Engineer': {
                'Programming and Software Development': 5,
                'Data Science and Analytics': 2,
                'Database Management': 3,
                'Cloud Computing': 9,
                'Project Management': 4,
                'Cybersecurity': 3,
                'IT Infrastructure and Networking': 10,
                'Artificial Intelligence and Machine Learning': 3,
                'System Administration': 9,
                'User Experience and Design': 2,
            },


            'Quality Assurance Engineer': {
                'Programming and Software Development': 6,
                'Data Science and Analytics': 2,
                'Database Management': 5,
                'Cloud Computing': 2,
                'Project Management': 3,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 7,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 2,
                'User Experience and Design': 2,
            },

            'Cloud Security Engineer': {
                'Programming and Software Development': 4,
                'Data Science and Analytics': 2,
                'Database Management': 2,
                'Cloud Computing': 10,
                'Project Management': 3,
                'Cybersecurity': 9,
                'IT Infrastructure and Networking': 6,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 4,
                'User Experience and Design': 2,
            },


            'IT Support Specialist': {
                'Programming and Software Development': 8,
                'Data Science and Analytics': 2,
                'Database Management': 2,
                'Cloud Computing': 2,
                'Project Management': 2,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 8,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 6,
                'User Experience and Design': 2,
            },


            'Business Analyst': {
                'Programming and Software Development': 3,
                'Data Science and Analytics': 7,
                'Database Management': 3,
                'Cloud Computing': 2,
                'Project Management': 6,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 2,
                'User Experience and Design': 2,
            },

            'Associate Software Engineer': {
                'Programming and Software Development': 10,
                'Data Science and Analytics': 4,
                'Database Management': 3,
                'Cloud Computing': 5,
                'Project Management': 4,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 4,
                'Artificial Intelligence and Machine Learning': 3,
                'System Administration': 3,
                'User Experience and Design': 4,
            },

            'Associate Data Scientist': {
                'Programming and Software Development': 4,
                'Data Science and Analytics': 7,
                'Database Management': 3,
                'Cloud Computing': 3,
                'Project Management': 3,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 7,
                'System Administration': 2,
                'User Experience and Design': 3,
            },

            'Associate Data Engineer': {
                'Programming and Software Development': 6,
                'Data Science and Analytics': 6,
                'Database Management': 7,
                'Cloud Computing': 4,
                'Project Management': 4,
                'Cybersecurity': 3,
                'IT Infrastructure and Networking': 3,
                'Artificial Intelligence and Machine Learning': 5,
                'System Administration': 4,
                'User Experience and Design': 3,
            },

            'Associate Machine Learning Engineer': {
                'Programming and Software Development': 7,
                'Data Science and Analytics': 8,
                'Database Management': 4,
                'Cloud Computing': 3,
                'Project Management': 3,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 10,
                'System Administration': 2,
                'User Experience and Design': 2,
            },

            'Associate Data Analyst': {
                'Programming and Software Development': 6,
                'Data Science and Analytics': 10,
                'Database Management': 7,
                'Cloud Computing': 3,
                'Project Management': 3,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 5,
                'System Administration': 2,
                'User Experience and Design': 2,
            },


            'Associate Database Administrator': {
                'Programming and Software Development': 6,
                'Data Science and Analytics': 3,
                'Database Management': 10,
                'Cloud Computing': 2,
                'Project Management': 5,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 7,
                'Artificial Intelligence and Machine Learning': 3,
                'System Administration': 8,
                'User Experience and Design': 2,
            },

            'Associate Project Manager': {
                'Programming and Software Development': 3,
                'Data Science and Analytics': 3,
                'Database Management': 3,
                'Cloud Computing': 4,
                'Project Management': 10,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 3,
                'Artificial Intelligence and Machine Learning': 3,
                'System Administration': 2,
                'User Experience and Design': 3,
            },

            'Associate Cyber Security Analyst': {
                'Programming and Software Development': 5,
                'Data Science and Analytics': 3,
                'Database Management': 3,
                'Cloud Computing': 4,
                'Project Management': 3,
                'Cybersecurity': 10,
                'IT Infrastructure and Networking': 3,
                'Artificial Intelligence and Machine Learning': 3,
                'System Administration': 2,
                'User Experience and Design': 3,
            },

            'Associate Network Engineer': {
                'Programming and Software Development': 2,
                'Data Science and Analytics': 2,
                'Database Management': 3,
                'Cloud Computing': 3,
                'Project Management': 2,
                'Cybersecurity': 6,
                'IT Infrastructure and Networking': 8,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 7,
                'User Experience and Design': 2,
            },

            'Associate UI/UX Designer': {
                'Programming and Software Development': 2,
                'Data Science and Analytics': 2,
                'Database Management': 2,
                'Cloud Computing': 2,
                'Project Management': 3,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 3,
                'System Administration': 2,
                'User Experience and Design': 10,
            },

            'Associate Cyber Security Engineer': {
                'Programming and Software Development': 4,
                'Data Science and Analytics': 8,
                'Database Management': 6,
                'Cloud Computing': 7,
                'Project Management': 4,
                'Cybersecurity': 10,
                'IT Infrastructure and Networking': 8,
                'Artificial Intelligence and Machine Learning': 6,
                'System Administration': 7,
                'User Experience and Design': 2,
            },


            'Associate Cloud Engineer': {
                'Programming and Software Development': 4,
                'Data Science and Analytics': 8,
                'Database Management': 3,
                'Cloud Computing': 10,
                'Project Management': 4,
                'Cybersecurity': 3,
                'IT Infrastructure and Networking': 8,
                'Artificial Intelligence and Machine Learning': 6,
                'System Administration': 7,
                'User Experience and Design': 2,
            },

            'Associate Quality Assurance Engineer': {
                'Programming and Software Development': 3,
                'Data Science and Analytics': 2,
                'Database Management': 2,
                'Cloud Computing': 2,
                'Project Management': 3,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 2,
                'User Experience and Design': 2,
            },

            'Associate Cloud Security Engineer': {
                'Programming and Software Development': 4,
                'Data Science and Analytics': 2,
                'Database Management': 2,
                'Cloud Computing': 7,
                'Project Management': 3,
                'Cybersecurity': 8,
                'IT Infrastructure and Networking': 7,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 4,
                'User Experience and Design': 2,
            },


            'Associate IT Support Specialist': {
                'Programming and Software Development': 3,
                'Data Science and Analytics': 6,
                'Database Management': 8,
                'Cloud Computing': 3,
                'Project Management': 4,
                'Cybersecurity': 7,
                'IT Infrastructure and Networking': 9,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 10,
                'User Experience and Design': 6,
            },

            'Associate Business Analyst': {
                'Programming and Software Development': 3,
                'Data Science and Analytics': 6,
                'Database Management': 3,
                'Cloud Computing': 2,
                'Project Management': 5,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 2,
                'User Experience and Design': 2,
            },

            'Intern Data Analyst': {
                'Programming and Software Development': 2,
                'Data Science and Analytics': 10,
                'Database Management': 2,
                'Cloud Computing': 2,
                'Project Management': 2,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 8,
                'System Administration': 2,
                'User Experience and Design': 2,
            },

            'Intern Data Engineer': {
                'Programming and Software Development': 6,
                'Data Science and Analytics': 9,
                'Database Management': 2,
                'Cloud Computing': 8,
                'Project Management': 2,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 8,
                'System Administration': 2,
                'User Experience and Design': 2,
            },

            'Intern Software Engineer': {
                'Programming and Software Development': 6,
                'Data Science and Analytics': 2,
                'Database Management': 2,
                'Cloud Computing': 3,
                'Project Management': 2,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 2,
                'User Experience and Design': 2,
            },

            'Intern Data Scientist': {
                'Programming and Software Development': 3,
                'Data Science and Analytics': 7,
                'Database Management': 3,
                'Cloud Computing': 2,
                'Project Management': 2,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 5,
                'System Administration': 2,
                'User Experience and Design': 2,
            },

            'Intern Project Manager': {
                'Programming and Software Development': 2,
                'Data Science and Analytics': 2,
                'Database Management': 2,
                'Cloud Computing': 3,
                'Project Management': 6,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 2,
                'User Experience and Design': 2,
            },

            'Intern Cloud Engineer': {
                'Programming and Software Development': 3,
                'Data Science and Analytics': 2,
                'Database Management': 2,
                'Cloud Computing': 6,
                'Project Management': 2,
                'Cybersecurity': 3,
                'IT Infrastructure and Networking': 6,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 4,
                'User Experience and Design': 2,
            },

            'Intern Database Administrator': {
                'Programming and Software Development': 2,
                'Data Science and Analytics': 3,
                'Database Management': 7,
                'Cloud Computing': 2,
                'Project Management': 2,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 5,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 4,
                'User Experience and Design': 2,
            },

            'Intern Network Engineer': {
                'Programming and Software Development': 2,
                'Data Science and Analytics': 2,
                'Database Management': 2,
                'Cloud Computing': 2,
                'Project Management': 2,
                'Cybersecurity': 5,
                'IT Infrastructure and Networking': 7,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 6,
                'User Experience and Design': 2,
            },


            'Intern Quality Assurance Engineer': {
                'Programming and Software Development': 2,
                'Data Science and Analytics': 2,
                'Database Management': 2,
                'Cloud Computing': 2,
                'Project Management': 2,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 2,
                'User Experience and Design': 2,
            },

            'Intern Cloud Security Engineer': {
                'Programming and Software Development': 2,
                'Data Science and Analytics': 2,
                'Database Management': 2,
                'Cloud Computing': 6,
                'Project Management': 2,
                'Cybersecurity': 7,
                'IT Infrastructure and Networking': 6,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 3,
                'User Experience and Design': 2,
            },

            'Intern UI/UX Designer': {
                'Programming and Software Development': 2,
                'Data Science and Analytics': 2,
                'Database Management': 2,
                'Cloud Computing': 2,
                'Project Management': 2,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 3,
                'System Administration': 2,
                'User Experience and Design': 6,
            },

            'Intern IT Support Specialist': {
                'Programming and Software Development': 7,
                'Data Science and Analytics': 6,
                'Database Management': 2,
                'Cloud Computing': 2,
                'Project Management': 7,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 10,
                'Artificial Intelligence and Machine Learning': 3,
                'System Administration': 10,
                'User Experience and Design': 6,
            },

            'Intern Business Analyst': {
                'Programming and Software Development': 2,
                'Data Science and Analytics': 5,
                'Database Management': 2,
                'Cloud Computing': 2,
                'Project Management': 4,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 2,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 2,
                'User Experience and Design': 2,
            },

            'Intern Cyber Security Analyst': {
                'Programming and Software Development': 3,
                'Data Science and Analytics': 8,
                'Database Management': 2,
                'Cloud Computing': 2,
                'Project Management': 4,
                'Cybersecurity': 10,
                'IT Infrastructure and Networking': 7,
                'Artificial Intelligence and Machine Learning': 2,
                'System Administration': 9,
                'User Experience and Design': 2,
            },

            'Intern Cyber Security Engineer': {
                'Programming and Software Development': 3,
                'Data Science and Analytics': 5,
                'Database Management': 4,
                'Cloud Computing': 2,
                'Project Management': 4,
                'Cybersecurity': 10,
                'IT Infrastructure and Networking': 6,
                'Artificial Intelligence and Machine Learning': 5,
                'System Administration': 8,
                'User Experience and Design': 4,
            },

            'Intern Machine Learning Engineer': {
                'Programming and Software Development': 7,
                'Data Science and Analytics': 9,
                'Database Management': 2,
                'Cloud Computing': 2,
                'Project Management': 4,
                'Cybersecurity': 2,
                'IT Infrastructure and Networking': 4,
                'Artificial Intelligence and Machine Learning': 10,
                'System Administration': 7,
                'User Experience and Design': 2,
            },

            
                                
        }


        # # Define the custom order for categories
        # custom_category_order = [
        #     'Programming and Software Development',
        #     'Data Science and Analytics',
        #     'Database Management',
        #     'Cloud Computing',
        #     'Project Management',
        #     'Cybersecurity',
        #     'IT Infrastructure and Networking',
        #     'Artificial Intelligence and Machine Learning',
        #     'System Administration',
        #     'User Experience and Design'
        # ]

        # # Convert the 'Category' column to a Categorical data type with the custom order
        # combined_category_totals['Category'] = pd.Categorical(combined_category_totals['Category'], categories=custom_category_order, ordered=True)

        # # Sort the DataFrame by the 'Category' column
        # combined_category_totals = combined_category_totals.sort_values(by='Category')

        # # Reset the index if needed
        # # combined_category_totals = combined_category_totals.reset_index(drop=True)

        # # Print the sorted DataFrame
        # print(combined_category_totals)


        # # Get the top 3 category totals
        # top_category_totals = category_totals.nlargest(3)

        # # Calculate the total sum of the top 3 category totals
        # total_sum_of_top_totals = top_category_totals.sum()

        # # Calculate the maximum possible sum (if all categories were in the top 3)
        # maximum_possible_sum = category_totals.sum()
 
        # # Calculate the final score as a percentage out of 100
        # ac_score = (total_sum_of_top_totals / maximum_possible_sum) * 100

        # # Print the final score (you can use it as needed)
        # print(f"Academic transcript Score: {ac_score:.2f}")


        # Generate the pie chart
        plt.figure(figsize=(12, 8))  # Adjust the size as needed

        plt.pie(
            category_totals,
            labels=None,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(edgecolor='w')  # Add white edgecolor to slices
        )

 

        # Set colors for each category
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

 

        # Customize colors for each pie slice
        for i, wedge in enumerate(plt.gca().patches):
            wedge.set_facecolor(colors[i % len(colors)])
            wedge.set_linewidth(1)  # Add border to slices


        # Add a legend with categories and colors
        legend_labels = category_totals.index

        plt.legend(
            legend_labels,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            title='Skill Areas',
            labelcolor='black',
            borderaxespad=0.5  # Adjust the distance of the legend from the chart
        )


        # Remove unnecessary spines and ticks
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().tick_params(axis='both', which='both', length=0)

 

        # Remove excess white space
        plt.subplots_adjust(left=0.0, right=0.75)

        # Save the pie chart to a file
        pie_chart_file = 'static/images/academic_transcript/pie_chart.png'
        plt.savefig(pie_chart_file, dpi=300)  # Adjust DPI as needed

 

        # Get candidate ID and name from the form
        candidate_id = request.form.get('candidateId')
        candidate_name = request.form.get('candidateName')
        position_applied = request.form.get('PositionApplied')

        # # Load the trained model
        # score_model = joblib.load('models/academic_transcript/RandomForestRegressor.joblib')


        # # Create a DataFrame with the 'Job role' and 'Weighted Grade' columns
        # data = {
        #     'Job role': [position_applied],  # Use the correct column name 'Job role' here
        #     'Programming and Software Development': [combined_category_totals.loc[combined_category_totals['Category'] == 'Programming and Software Development', 'Weighted Grade'].values[0]],
        #     'Data Science and Analytics': [combined_category_totals.loc[combined_category_totals['Category'] == 'Data Science and Analytics', 'Weighted Grade'].values[0]],
        #     'Database Management': [combined_category_totals.loc[combined_category_totals['Category'] == 'Database Management', 'Weighted Grade'].values[0]],
        #     'Cloud Computing': [combined_category_totals.loc[combined_category_totals['Category'] == 'Cloud Computing', 'Weighted Grade'].values[0]],
        #     'Project Management': [combined_category_totals.loc[combined_category_totals['Category'] == 'Project Management', 'Weighted Grade'].values[0]],
        #     'Cybersecurity': [combined_category_totals.loc[combined_category_totals['Category'] == 'Cybersecurity', 'Weighted Grade'].values[0]],
        #     'IT Infrastructure and Networking': [combined_category_totals.loc[combined_category_totals['Category'] == 'IT Infrastructure and Networking', 'Weighted Grade'].values[0]],
        #     'Artificial Intelligence and Machine Learning': [combined_category_totals.loc[combined_category_totals['Category'] == 'Artificial Intelligence and Machine Learning', 'Weighted Grade'].values[0]],
        #     'System Administration': [combined_category_totals.loc[combined_category_totals['Category'] == 'System Administration', 'Weighted Grade'].values[0]],
        #     'User Experience and Design': [combined_category_totals.loc[combined_category_totals['Category'] == 'User Experience and Design', 'Weighted Grade'].values[0]]
        # }

        # data_df = pd.DataFrame(data)
        # print(data_df)

        # # Perform one-hot encoding for 'Job role'
        # data_df = pd.get_dummies(data_df, columns=['Job role'])

        # # Make predictions using the trained model
        # predicted_score = score_model.predict(data_df)

        # # Print the predicted score
        # print("Predicted Score:", predicted_score[0])

        # Check if the position_applied exists in the job_title_to_skill_weights dictionary
        if position_applied in job_title_to_skill_weights:
            # Get the skill weights for the specified job title
            skill_weights = job_title_to_skill_weights[position_applied]

            # Initialize a variable to store the total score
            total_score = 0

            # Calculate the sum of all weighted grades for the categories
            total_weighted_grade_sum = combined_category_totals['Weighted Grade'].sum()


            # Iterate through the categories and calculate the score
            for category, weighted_grade in zip(combined_category_totals['Category'], combined_category_totals['Weighted Grade']):
                if category in skill_weights:
                    skill_weight = skill_weights[category]
                    # Calculate the percentage for the category based on the sum of all weighted grades
                    category_percentage = (weighted_grade / total_weighted_grade_sum) * 100

                    category_score = category_percentage * skill_weight
                    print(category, category_score)
                    total_score += category_score

            
            
            # Calculate the maximum possible score for this position
            max_possible_score = 1000

            # Calculate the score as a percentage
            ac_score1 = (total_score / max_possible_score) * 100

            # Round off ac_score to 2 decimal places
            ac_score = round(ac_score1, 2)


            session['ac_score'] = ac_score

            # Print or return the total score
            print("Total Score for", position_applied, ":", ac_score)
        else:
            print("Job title not found in skill weights dictionary")


        return render_template('academic_transcript/AcaedmicTranscriptsResults.html', pie_chart=pie_chart_file, candidate_id=candidate_id, candidate_name=candidate_name,  position_applied=position_applied, ac_score=ac_score)

    return render_template('academic_transcript/AcaedmicTranscriptsIndex.html', error='Please upload a PDF file.')

#Academic Transcript - Shanali - END -----------------------------------------------------------------------------------------------------------

# Final score - START

@app.route('/final_score')
def final_score():
    return render_template('final_score.html')


@app.route('/calc_final_score', methods=['POST', 'GET'])
def calcFinalScore():

    cand_name = session.get('cand_name')
    jobrole = session.get('jobrole')

# CV--------------------------------------------------------------------------------
    # cv_ranking = session.get('ranking')
    
    # matching_percentages = [result['matching_percentage'] for result in cv_ranking]
    # highest_matching_percentage = max(matching_percentages)

    # # Find the entry with the highest matching_percentage 
    # entry_with_highest_percentage = max( cv_ranking, key=lambda x: x["matching_percentage"] ) 
    # #Get the skills from the entry with the highest matching_percentage 
    # skills_with_highest_percentage = entry_with_highest_percentage.get("skills", [])

    # # Convert the list to a set to get unique words
    # unique_words = set(skills_with_highest_percentage)

    # # Convert the set back to a list if needed
    # unique_words_list = list(unique_words)

# GITHUB-------------------------------------------------------------
    percentage_scores = session.get('percentage_scores')

    # Initialize an empty list to store the words
    plp_words = []
    # Loop through the keys of the dictionary
    for key in percentage_scores.keys():
        # Split the key into words using whitespace as the delimiter
        key_words = key.split()
        # Extend the list of words with the words from the current key
        plp_words.extend(key_words)

    #cv-gitHub technical skills validation-----------------------
    github_plp_set = set(plp_words)
    cv_plp_set = set(unique_words_list)
    # Find the common words
    common_words = github_plp_set.intersection(cv_plp_set)

    # Calculate the percentage of common words
    percentage_common = (len(common_words) / (len(github_plp_set) + len(cv_plp_set))) * 100

#LinkedIn Predicted Job Category-------------------------------------------
    predicted_category = session.get('predicted_category')
#Recommendation Positive sentiment-----------------------------------------
    positive_percentage = session.get('positive_percentage')
# PERSONALITY-----------------------------------------------------------------------------------------

    personality_score = session.get('personality_score')

# ACADEMIC TRANSCRIPT---------------------------------------------------------------------------------

    # #Academic transcript score
    ac_score= session.get('ac_score')

    #Without sandani's
    return render_template('final_score.html', cand_name=cand_name, jobrole=jobrole, personality_score=personality_score, highest_matching_percentage=highest_matching_percentage, ac_score=ac_score, textarea_content="", slider_values="")

    # With sandani's , once the token issue is solved.
    return render_template('final_score.html', cand_name=cand_name, jobrole=jobrole, common_words=common_words, personality_score=personality_score, ac_score=ac_score, highest_matching_percentage=highest_matching_percentage, percentage_common=percentage_common, textarea_content="", slider_values="")

if __name__ == '__main__':
    app.run(debug=True)
