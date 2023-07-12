from flask import Flask

app = Flask(__name__)

# Members API Route
@app.route("/personality")
def personality():
    return {"prediction": "Personality Prediction"}

if __name__ == "__main__":
    app.run(debug=True)
