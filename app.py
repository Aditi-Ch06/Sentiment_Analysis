# import pickle
# from flask import Flask, jsonify, request
# import pandas as pd
# import numpy as np
# import nltk
# from nltk.corpus import stopwords
# import re
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# app = Flask(__name__)

# # loading trained model
# model = pickle.load(open('sentiment_analysis_model.pkl','rb'))
# encoder = pickle.load(open('encoder.pkl', 'rb'))

# # Tokenize text
# max_words = 10000  # Vocabulary size
# max_sequence_len = 100  # Maximum length of input sequences
# tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
# nltk.download('stopwords')
# def clean_text(text):
#         text = text.lower()
#         text = re.sub(r'[^\w\s]', '', text) # remove punctuation
#         text = re.sub(r'\d+', '', text) # remove numbers
#         text = re.sub(r'\s+', ' ', text) # remove extra spaces
#         return text
# def remove_stopwords(text):
#         return ' '.join(word for word in text.split() if word not in stopwords.words('english'))
# def predict_sentiment(text):
#         cleaned_text = clean_text(text)
#         cleaned_text = remove_stopwords(cleaned_text)

#         sequence = tokenizer.texts_to_sequences([cleaned_text])
#         padded_sequence = pad_sequences(sequence, maxlen=max_sequence_len, padding='post')
        
#         prediction = model.predict(padded_sequence)
#         sentiment = encoder.inverse_transform([np.argmax(prediction)])
#         return sentiment[0]


# # @app.route('/predict', methods=['POST'])

# # def predict():
# #     #  input data
# #     input_data = request.get_json()
# #     if not input_data :
# #          return jsonify("error : No data received")

# #     # # coverting into df
# #     # df = pd.DataFrame(input_data)

# #     # # pre-processing before passing to model

# #     # df['Answer'] = df['Answer'].astype(str)
# #     # # lower-casing
# #     # df['Answer'] = df['Answer'].str.lower()
# #     # # removing special characters(punctuations, special symbols, and numbers that don't contribute to sentiment)
    
# #     # df['Answer'] = df['Answer'].apply(clean_text)
# #     # # removing stop words(like "is", "the", "and" might dilute signal for sentiment) using nltk
    
    
# #     # df['Answer'] = df['Answer'].apply(remove_stopwords)
    
# #     # tokenizer.fit_on_texts(df['Answer'])
# #     # sequences = tokenizer.texts_to_sequences(df['Answer'])
# #     # # Pad sequences
# #     # X = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

    
# #     # prediction 
# #     print(f"Received input: {input_data}")  # Debugging line

# #     output = predict_sentiment(input_data['text'])
# #     return jsonify(f"Predicted Sentiment : {output}")

# @app.route('/predict', methods=['POST'])
# def predict():
#     input_data = request.get_json()
#     if not input_data:
#         return jsonify("error: No data received")
    
#     output = predict_sentiment(input_data['text'])
#     return jsonify(f"Predicted Sentiment: {output}")


# if __name__=='__main__':
#  app.run(debug=True)

import pickle
from flask import Flask, jsonify, request
from keras.models import load_model
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# loading trained model
# model = pickle.load(open('sentiment_analysis_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
model = load_model('sentiment_analysis_model.h5')
with open('tokenizer.pkl','rb') as f:
    tokenizer = pickle.load(f)


# Tokenize text
max_words = 10000  # Vocabulary size
max_sequence_len = 100  # Maximum length of input sequences
# tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text

def remove_stopwords(text):
    return ' '.join(word for word in text.split() if word not in stopwords.words('english'))

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    print(f"Cleaned text: {cleaned_text}")
    cleaned_text = remove_stopwords(cleaned_text)
    print(f"Text after stopword removal: {cleaned_text}")

    # tokenizer.fit_on_texts([cleaned_text])
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    print(f"Tokenized sequence: {sequence}") 

    if not sequence or not sequence[0]:  # Check if sequence is empty
        print("Error: The tokenized sequence is empty!")
        return "Error: Invalid input text"
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_len, padding='post')
    print(f"Padded sequence: {padded_sequence}")
    prediction = model.predict(padded_sequence)
    sentiment = encoder.inverse_transform([np.argmax(prediction)])
    return sentiment[0]

@app.route('/', methods=["GET"])
def home():
    return jsonify({"message": "Flask app is running!"})


@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        print("request received")
        input_data = request.get_json()
        if not input_data:
            return jsonify("error: No data received")
        if "text" not in input_data:
            return jsonify("error: 'text' field is required")
        # Hardcoded test data to see if the issue is with input processing
        test_text = input_data.get("text", "I love this app!")
        print(f"Using text: {test_text}")  # Debugging statement
        
        output = predict_sentiment(test_text)
        return jsonify({
            'predicted_sentiment': output
        })
    except Exception as e:
        return jsonify({'error':str(e)})

if __name__ == '__main__':
    app.run(debug=True)