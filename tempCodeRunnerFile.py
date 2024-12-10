@app.route('/', methods=["GET"])
def home():
    return jsonify({"message": "Flask app is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    print("request received")
    input_data = request.get_json()

    if not input_data:
        return jsonify("error: No data received")
    
    if "text" not in input_data:
        return jsonify("error: 'text' field is required")

    output = predict_sentiment(input_data['text'])
    return jsonify(f"Predicted Sentiment: {output}")