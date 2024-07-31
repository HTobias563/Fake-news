import time
import torch
import pandas as pd 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim import AdamW
from transformers import DistilBertModel, DistilBertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Load and preprocess data
df_neu = pd.read_csv("data/WELFake_Dataset_preprocessed_final_cleaned.csv")
df_neu = df_neu.fillna('') 

MAX_SEQUENCE_LENGTH = 250
SEED = 10
np.random.seed(SEED)
torch.manual_seed(SEED)

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Split data into train, validation, and test sets
texts = df_neu['text'].tolist()
titles = df_neu['title'].tolist()
labels = df_neu['label'].tolist()

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

encoded_train_texts, encoded_train_titles = tokenize_data(train_texts, train_titles, tokenizer, MAX_SEQUENCE_LENGTH)
encoded_val_texts, encoded_val_titles = tokenize_data(val_texts, val_titles, tokenizer, MAX_SEQUENCE_LENGTH)
encoded_test_texts, encoded_test_titles = tokenize_data(test_texts, test_titles, tokenizer, MAX_SEQUENCE_LENGTH)

# Extract inputs and attention masks
train_input_ids_texts = encoded_train_texts['input_ids']
train_attention_masks_texts = encoded_train_texts['attention_mask']
train_input_ids_titles = encoded_train_titles['input_ids']
train_attention_masks_titles = encoded_train_titles['attention_mask']

val_input_ids_texts = encoded_val_texts['input_ids']
val_attention_masks_texts = encoded_val_texts['attention_mask']
val_input_ids_titles = encoded_val_titles['input_ids']
val_attention_masks_titles = encoded_val_titles['attention_mask']

test_input_ids_texts = encoded_test_texts['input_ids']
test_attention_masks_texts = encoded_test_texts['attention_mask']
test_input_ids_titles = encoded_test_titles['input_ids']
test_attention_masks_titles = encoded_test_titles['attention_mask']

# Convert labels to tensors
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
test_labels = torch.tensor(test_labels)

train_dataset = TensorDataset(train_input_ids_texts, train_attention_masks_texts, train_input_ids_titles, train_attention_masks_titles, train_labels)
val_dataset = TensorDataset(val_input_ids_texts, val_attention_masks_texts, val_input_ids_titles, val_attention_masks_titles, val_labels)
test_dataset = TensorDataset(test_input_ids_texts, test_attention_masks_texts, test_input_ids_titles, test_attention_masks_titles, test_labels)

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# Initialize model
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

model = DistilBERTClassifier()

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Added weight decay for regularization
epochs = 15 # Set a higher number of epochs to leverage early stopping
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Early stopping parameters
patience = 3  # Number of epochs to wait for improvement before stopping
best_loss = float('inf')
patience_counter = 0


# Track training and validation loss and accuracy
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Training loop
start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    
    for step, batch in enumerate(train_loader):
        batch_input_ids_texts, batch_attention_masks_texts, batch_input_ids_titles, batch_attention_masks_titles, batch_labels = batch
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=batch_input_ids_texts,
            attention_mask=batch_attention_masks_texts
        )
        
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs, batch_labels)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train += batch_labels.size(0)
        correct_train += (predicted == batch_labels).sum().item()
    
    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    
    # Evaluation on validation set after each epoch
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            batch_input_ids_texts, batch_attention_masks_texts, batch_input_ids_titles, batch_attention_masks_titles, batch_labels = batch
            
            outputs = model(
                input_ids=batch_input_ids_texts,
                attention_mask=batch_attention_masks_texts
            )
            
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, batch_labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += batch_labels.size(0)
            correct_val += (predicted == batch_labels).sum().item()

            predictions.append(predicted)
            true_labels.append(batch_labels)
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy*100:.2f}%')
    
    # Early stopping check
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'distilbert_model.pt')
        print('Model saved as distilbert_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping after epoch {epoch+1}')
            break

end_time = time.time()
training_time = end_time - start_time
print(f'Total training time: {training_time / 60:.2f} minutes for {len(train_loader.dataset)} samples')

# Plotting the training and validation loss and accuracy
def plot_graphs(train_metric, val_metric, metric_name):
    plt.plot(train_metric)
    plt.plot(val_metric, '')
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend([metric_name, 'val_' + metric_name])

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plot_graphs(train_accuracies, val_accuracies, 'accuracy')
plt.subplot(1, 2, 2)
plot_graphs(train_losses, val_losses, 'loss')
plt.show()

# Evaluation on the test set
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
