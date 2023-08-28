from flask import Flask, render_template, request
import pickle
from ScrapeLinkedIn import scrape_linkedin_skills

app = Flask(__name__)

# Load the machine learning model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        linkedin_profile_url = request.form['linkedin_profile_url']
        skills = scrape_linkedin_skills(linkedin_profile_url)  # Calling scraping function
        if skills:
            predicted_category = model.predict([skills])  #model takes a list of skills
           # predicted_category = model.predict(fitted_vectorizer.transform([skills]))
            result = f"Predicted Job Category: {predicted_category[0]}"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
