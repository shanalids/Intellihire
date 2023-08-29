from flask import Flask, render_template, request
from github import Github
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

def calculate_language_proficiency(username, access_token):
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

def generate_pie_chart(percentage_scores):
    languages = list(percentage_scores.keys())
    percentages = list(percentage_scores.values())
    
    fig, ax = plt.subplots()
    ax.pie(percentages, labels=languages, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.axis('equal')
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
    return render_template('plp_form.html')

@app.route('/plp', methods=['GET', 'POST'])
def plp():
    if request.method == 'POST':
        username = request.form.get('username')
        access_token = "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"
        
        percentage_scores = calculate_language_proficiency(username, access_token)
        pie_chart = generate_pie_chart(percentage_scores)
        
        return render_template('plp.html', username=username, percentage_scores=percentage_scores, pie_chart=pie_chart)
    
    return render_template('plp.html', username=None, percentage_scores=None, pie_chart=None)



if __name__ == '__main__':
    app.run(debug=True)
