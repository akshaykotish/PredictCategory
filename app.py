import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import custom_object_scope
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Define a custom transformer layer
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Load the data to initialize tokenizer and label encoder
try:
    data = pd.read_csv('Complaint_Dataset.csv', encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv('Complaint_Dataset.csv', encoding='latin1')  # or try 'iso-8859-1'

# Initialize tokenizer and label encoder
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Description'])

label_encoder = LabelEncoder()
label_encoder.fit(data['Category'])

# Register the custom object
tf.keras.utils.get_custom_objects().update({'TransformerBlock': TransformerBlock})

app = Flask(__name__)

# Load the pre-trained model with custom object scope
with custom_object_scope({'TransformerBlock': TransformerBlock}):
    model = tf.keras.models.load_model('loan_fraud_model.h5')

def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=306, padding='post')  # Adjust maxlen as needed
    return padded_sequence

def predict_category(text):
    processed_text = preprocess_text(text)
    predictions = model.predict(processed_text)
    predicted_category = label_encoder.inverse_transform([np.argmax(predictions, axis=1)[0]])[0]
    return predicted_category

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    category = predict_category(text)
    return jsonify({'category': category})

if __name__ == '__main__':
    app.run(debug=True)
