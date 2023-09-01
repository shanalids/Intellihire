from flask import Flask, request, render_template, Response
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from github import Github

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('compare_form.html')

@app.route('/compare', methods=['GET', 'POST'])
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

        return render_template('compare.html', img_base64=img_base64)

    return render_template('compare.html')

if __name__ == '__main__':
    app.run(debug=True)
