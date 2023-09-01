from flask import Flask, render_template, request
import pickle
from ScrapeLinkedIn import scrape_linkedin_skills

app = Flask(__name__)

# # Load the machine learning model
# with open('model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# # Load the fitted vectorizer
# with open('fitted_vectorizer.pkl', 'rb') as file:
#     fitted_vectorizer = pickle.load(file)
model = pickle.load(open("model.pkl", "rb"))
fitted_vectorizer = pickle.load(open("fitted_vectorizer.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        linkedin_profile_url = request.form['linkedin_profile_url']
        skills = scrape_linkedin_skills(linkedin_profile_url)  # Calling scraping function
        if skills:
            #predicted_category = model.predict([skills])  #model takes a list of skills
            predicted_category = model.predict(fitted_vectorizer.transform(skills))
            result = f"Predicted Job Category: {predicted_category[0]}"
    return render_template('job_cat_form.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
