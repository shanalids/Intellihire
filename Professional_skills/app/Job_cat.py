from flask import Flask, render_template, request
import pickle
import requests
# from ScrapeLinkedIn import scrape_linkedin_skills

app = Flask(__name__)

def scrape_linkedin_skills(linkedin_profile_url):
    api_key = 'MxVwlMuCI00hrmsugxWLjA'
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
