import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import io
import base64

app = Flask(__name__)

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

@app.route('/')
def index1():
    return render_template('sentiment.html')


@app.route('/sentiment_results', methods=['GET', 'POST'])
def index2():
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
            
            # Update sentiment analysis and visualization function to return column names
            column_names = last_row.index.tolist()
            plot_data = analyze_sentiment_and_visualize(last_row)
            row_content = last_row.values.tolist()

    return render_template('sentiment_results.html', plot_data=plot_data, column_names=column_names, row_content=row_content)




@app.route('/sentiment_results', methods=['GET', 'POST'])
def index3():
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

    return render_template('sentiment_results.html', plot_data=plot_data)

if __name__ == '__main__':
    app.run(debug=True)
