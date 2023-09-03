from flask import Flask, render_template, request, session
import joblib
from tabulate import tabulate
import pandas as pd
import nltk
from process import *
import numpy as np
import pickle
import file_new as dp
from helper import *
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

app = Flask(__name__)
app.secret_key = 'cv-analysis'

model = pickle.load(open("models\stkmodel.pkl", "rb"))
saved_filename = "models\Vectorizer1.joblib"
vectorizer = joblib.load(saved_filename)

# Initialize a dictionary to store session data


 # Initialize Flask-Session

@app.route('/ranking', methods=['GET', 'POST'])
def index():
    session_data = {}  # Define session_data as an empty dictionary
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
            jd_text = preprocess_text(jd)
            keywords_jd = dp.spacy_keywords(jd_text)

            #skills_list = ["Python", "Java", "Machine Learning", "Data Analysis", "Communication"]

            matching_results = []

            for uploaded_pdf in uploaded_pdfs:
                if uploaded_pdf.filename.endswith('.pdf'):
                    # Preprocessing of resume
                    pdf = extract_pdf_text(uploaded_pdf)
                    pdf_text = preprocess_text(pdf)
                    
                    
                    # Keyword extraction of resume
                    keywords_resume = dp.nltk_keywords(pdf_text)

                    # Start matching prediction
                    new_data_transformed = vectorizer.transform([pdf_text + " " + jd_text])
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
                    print(skills_list)
                    extracted_skills_frequency = extract_skills_and_count_frequency(pdf, skills_list)
                    skills=extract_skills(pdf,skills_list)
                    import json

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
            return render_template('cv_analysis/results.html', ranking=ranking)

    return render_template('cv_analysis/index.html', ranking=[])




@app.route('/profile/<int:pdf_index>')
def view_profile(pdf_index):
    if 'matching_results' in session:
        matching_results = session['matching_results']
        
        
        # Check if the selected PDF index is within the valid range
        if pdf_index >= 0 and pdf_index < len(matching_results):
            selected_profile = matching_results[pdf_index]
            return render_template('CV_analysis/profile.html', profile=selected_profile)
    
    # Handle the case when the PDF index is invalid or matching results are not available
    return "Profile not found", 404


def extract_skills(resume_text, skills_list):
    # Create a regex pattern to match skills (case-insensitive)
    skills_pattern = r'\b(?:' + '|'.join(re.escape(skill.strip()) for skill in skills_list.split(',')) + r')\b'
    
    # Find all matches of skills in the resume text
    matches = re.findall(skills_pattern, resume_text, flags=re.IGNORECASE)
    
    return matches


import re
def extract_skills_and_count_frequency(resume_text, skills_list):
    # Create a regex pattern to match skills (case-insensitive)
    skills_pattern = r'\b(?:' + '|'.join(re.escape(skill.strip()) for skill in skills_list.split(',')) + r')\b'
    
    # Find all matches of skills in the resume text
    matches = re.findall(skills_pattern, resume_text, flags=re.IGNORECASE)
    
    # Count the frequency of each skill
    skill_frequency = Counter(matches)
    
    return skill_frequency






if __name__ == '__main__':
    app.run(debug=True)

