Sure, here's a complete `README.md` file incorporating all the details:

### `README.md`
```markdown
# PredictCategory

This repository contains a Flask web application that uses a pre-trained TensorFlow model to predict the category of crime based on text descriptions. The application provides real-time predictions and a user-friendly interface.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Screenshots](#screenshots)
- [Model Training](#model-training)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/akshaykotish/PredictCategory.git
   cd PredictCategory
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the necessary data file (`your_data.csv`) in the root directory.

5. Run the Flask application:

   ```bash
   python app.py
   ```

6. Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Usage

1. Open the application in your web browser.
2. Enter a description of a crime in the textarea provided.
3. The predicted category will appear below the textarea in real-time as you type.

## Example

### Fraud Example

```text
The victim received an email from someone claiming to be a loan officer from a reputable bank. The email stated that the victim was pre-approved for a low-interest loan and requested personal information to proceed with the application. The victim provided their Social Security number, bank account details, and other personal information. Shortly after, unauthorized transactions appeared in the victim's bank account, and the victim realized their identity had been stolen.
```

## Screenshots

![Main Page](./Screenshot%20(272).png)
*Main interface of the application where users can enter text descriptions.*

![Prediction Example](./Screenshot%20(273).png)
*Example prediction showing the category of a crime based on the description.*

## Model Training

The model was trained using the following script:

```python
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data with a specified encoding
try:
    data = pd.read_csv('your_data.csv', encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv('your_data.csv', encoding='latin1')  # or try 'iso-8859-1'

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Description'])

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(data['Description'])
padded_sequences = pad_sequences(sequences, padding='post')

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['Category'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout, GlobalAveragePooling1D, Input
from tensorflow.keras.models import Model

# Define a custom transformer layer
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define the model
embed_dim = 64  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer

inputs = Input(shape=(X_train.shape[1],))
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embed_dim)(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(embedding_layer)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(len(label_encoder.classes_), activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save the model with custom objects
model.save('loan_fraud_model.h5', save_format='h5', include_optimizer=True)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

Created by Akshay Kotish & Co.
```

### Notes

1. **Installation Instructions**: Detailed steps to set up the environment and run the application.
2. **Usage Instructions**: How to use the application.
3. **Example**: A sample fraud description for testing.
4. **Screenshots**: Include placeholders for screenshots (you need to capture the screenshots and place them in a `screenshots` directory).
5. **Model Training Script**: Included the script used for training the model for transparency and reproducibility.
6. **License**: Placeholder for licensing information.
7. **Authors**: Credited the creators.

Remember to capture and add the screenshots to the `screenshots` directory, and update the image paths in the `README.md` file accordingly.