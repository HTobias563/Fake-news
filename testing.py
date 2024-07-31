import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the model class (must match the original model definition)
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

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the saved model
model = DistilBERTClassifier()
model.load_state_dict(torch.load('distilbert_model.pt'))
model.eval()  # Set the model to evaluation mode

# Load and preprocess the dataset
df_neu = pd.read_csv("data/WELFake_Dataset_preprocessed_final_cleaned.csv")
df_neu = df_neu.fillna('')

# Prepare the dataset splits
texts = df_neu['text'].tolist()
titles = df_neu['title'].tolist()
labels = df_neu['label'].tolist()

# Split the data into train, validation, and test sets
train_texts, test_texts, train_titles, test_titles, train_labels, test_labels = train_test_split(
    texts, titles, labels, test_size=0.2, random_state=SEED
)

val_texts, test_texts, val_titles, test_titles, val_labels, test_labels = train_test_split(
    test_texts, test_titles, test_labels, test_size=0.5, random_state=SEED
)

# Tokenize data
def tokenize_data(texts, titles, tokenizer, max_length):
    encoded_texts = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    encoded_titles = tokenizer(
        titles,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    return encoded_texts, encoded_titles

encoded_texts, encoded_titles = tokenize_data(test_texts, test_titles, tokenizer, 250)

input_ids_texts = encoded_texts['input_ids']
attention_masks_texts = encoded_texts['attention_mask']
input_ids_titles = encoded_titles['input_ids']
attention_masks_titles = encoded_titles['attention_mask']

# Convert labels to tensors
labels = torch.tensor(test_labels)

# Create TensorDataset
test_dataset = TensorDataset(input_ids_texts, attention_masks_texts, input_ids_titles, attention_masks_titles, labels)

batch_size = 16
test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# Evaluate the model
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        batch_input_ids_texts, batch_attention_masks_texts, batch_input_ids_titles, batch_attention_masks_titles, batch_labels = batch
        
        outputs = model(
            input_ids=batch_input_ids_texts,
            attention_mask=batch_attention_masks_texts
        )
        
        _, predicted = torch.max(outputs.data, 1)
        predictions.append(predicted)
        true_labels.append(batch_labels)

predictions = torch.cat(predictions, dim=0)
true_labels = torch.cat(true_labels, dim=0)

# Convert predictions to class labels
predicted_labels = predictions

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f'Final Accuracy: {accuracy * 100:.2f}%')

# Print classification report
report = classification_report(true_labels, predicted_labels, digits=4)
print('Classification Report:')
print(report)

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print('Confusion Matrix:')
print(conf_matrix)

# Calculate F1 score
f1 = f1_score(true_labels, predicted_labels, average='weighted')
print(f'F1 Score: {f1:.4f}')
