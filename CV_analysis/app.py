from flask import Flask, render_template, request
import fitz  # PyMuPDF
import re
import joblib
from joblib import load
import nltk
from helper import *
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import nltk


model = pickle.load(open("models\stkmodel.pkl", "rb"))

saved_filename = "models\Vectorizer1.joblib"
vectorizer1 = joblib.load(saved_filename)
app = Flask(__name__)

# Initialize the vectorizer
#vectorizer1 = CountVectorizer()

@app.route('/', methods=['GET', 'POST'])
def index():
    pdf_text = ""
    jd_text = ""
    new_data_transformed = ""
    new_data_prediction = ""
    if request.method == 'POST':
        uploaded_pdf = request.files.get('pdf')
        uploaded_jd = request.files.get('jd')

        if uploaded_pdf and uploaded_pdf.filename.endswith('.pdf'):
            pdf_text = preprocess_text(extract_pdf_text(uploaded_pdf))
        if uploaded_jd and uploaded_jd.filename.endswith('.pdf'):
            jd_text = preprocess_text(extract_pdf_text(uploaded_jd))

        # Transform new data using the fitted vectorizer
        new_data_transformed = vectorizer1.transform([pdf_text + " " + jd_text])
        new_data_prediction = model.predict(new_data_transformed)
    return render_template('index.html',new_data_prediction=new_data_prediction)

if __name__ == '__main__':
    app.run(debug=True)
