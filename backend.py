from flask import Flask, request, jsonify

from test_model import make_prediction

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    return jsonify(make_prediction([text]))

if __name__ == '__main__':
    app.run(debug=True)
