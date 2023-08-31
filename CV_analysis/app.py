from flask import Flask, render_template, request
import joblib
from tabulate import tabulate
import pandas as pd
import nltk
import pickle
import file_new as dp
from helper import *
from sklearn.feature_extraction.text import CountVectorizer
from urllib.parse import quote_plus

app = Flask(__name__)

model = pickle.load(open("models\stkmodel.pkl", "rb"))
saved_filename = "models\Vectorizer1.joblib"
vectorizer = joblib.load(saved_filename)



@app.route('/ranking', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_pdfs = request.files.getlist('pdf')
        uploaded_jd = request.files.get('jd')

        jd_text = " "
        jd = " "

        if uploaded_jd and uploaded_jd.filename.endswith('.pdf'):
            jd = extract_pdf_text(uploaded_jd)
            jd_text = preprocess_text(jd)

            keywords_jd = dp.spacy_keywords(jd_text)
            print("keywords in jd", keywords_jd)

            matching_results = []

            for uploaded_pdf in uploaded_pdfs:
                if uploaded_pdf.filename.endswith('.pdf'):
                    pdf = extract_pdf_text(uploaded_pdf)
                    pdf_text = preprocess_text(pdf)
                    keywords_resume = dp.nltk_keywords(pdf_text)

                    new_data_transformed = vectorizer.transform([pdf_text + " " + jd_text])
                    new_data_prediction = model.predict(new_data_transformed)

                    jd_keywords_in_resume_table = []
                    matching_keywords=[]
                    for word in keywords_jd:
                        if word in keywords_resume:
                            match_result = [word, 'Match']
                            matching_keywords.append(word)
                        else:
                            match_result = [word, 'No Match']
                        jd_keywords_in_resume_table.append(match_result)

                    #jd_keywords_in_resume_table = tabulate(jd_keywords_in_resume_table, headers=['Words', 'Keywords in JD', 'JD-Resume Match Result'], showindex='always', tablefmt='grid')
                    
                    # Convert the list of dictionaries into a DataFrame
                    

                    print(f'Comparing Resume and Job Description for PDF {uploaded_pdf.filename}:')
                    print(jd_keywords_in_resume_table)

                    matching_results.append({
                        "pdf_filename": uploaded_pdf.filename,
                        "matching_percentage": new_data_prediction[0],
                        "pdf_text": pdf_text,
                        "jd_keywords_in_resume_table":jd_keywords_in_resume_table,
                        "matching_keywords":matching_keywords
                    })
                    
            # Rank matching results and render the template
            ranking = sorted(matching_results, key=lambda x: x["matching_percentage"], reverse=True)
            return render_template('CV_analysis/results.html', ranking=ranking)

    return render_template('CV_analysis/index.html', ranking=[])




@app.route('/CV_analysis/profile/<filename>/<float:percentage>/<text>/<table>/<match>/', methods=['GET'])
def profile(filename, percentage, text,table,match):
    
    return render_template('CV_analysis/profile.html', pdf_filename=filename, matching_percentage=percentage, pdf_text=text,jd_keywords_in_resume_table=table,matching_keywords=match)

if __name__ == '__main__':
    app.run(debug=True)
