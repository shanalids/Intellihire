from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('kmeans_model.pkl', 'rb'))

# Members API Route
@app.route('/')
def personality():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    output = model.predict([[4,3,2,1,3,3,2,2,1,4,3,5,5,4,5,5,5,4,4,4,5,3,3,4,5,4,3,2,1,3,3,2,2,1,4,3,5,5,4,5,5,5,4,4,4,5,3,3,4,5]])
    return render_template('index.html', prediction_text='Personality cluster is {}' .format(output))



if __name__ == "__main__":
    app.run(debug=True)
