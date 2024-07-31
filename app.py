from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer

app = Flask(__name__)

class DistilBERTClassifier(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(DistilBERTClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, 2)  # Assuming binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]  # Use first token representation
        pooled_output = self.dropout(hidden_state)
        logits = self.classifier(pooled_output)
        return logits

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBERTClassifier()
model.load_state_dict(torch.load('distilbert_model.pt'))
model.eval()  # Set the model to evaluation mode

def predict(text, title, tokenizer, model):
    # Tokenize the input text
    encoded_texts = tokenizer(
        [text],
        max_length=250,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    encoded_titles = tokenizer(
        [title],
        max_length=250,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids_texts = encoded_texts['input_ids']
    attention_masks_texts = encoded_texts['attention_mask']
    input_ids_titles = encoded_titles['input_ids']
    attention_masks_titles = encoded_titles['attention_mask']

    # Make prediction
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_texts,
            attention_mask=attention_masks_texts
        )
        predicted = torch.argmax(outputs, dim=1).item()
    
    if predicted == 1:
        return "Fake"
    else:
        return "Real"

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict_route():
    title = request.form.get("Articel_titel")
    text = request.form.get("Articel_text")
    prediction = predict(text, title, tokenizer, model)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
