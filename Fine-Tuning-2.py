#Python Example: Fine-Tuning DistilBERT on IMDB Reviews
# python3 -m venv .vfinetuning
# source .vfinetuning/bin/activate
# python3 Fine-Tuning-2.py

# Install required packages if not installed
# pip install datasets
# pip install torch
# pip install transformers 
# pip install scikit-learn
# pip install "transformers[torch]" "accelerate>=1.12.0" -U
# pip install "numpy<2"


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EarlyStoppingCallback
import matplotlib.pyplot as plt
import pandas as pd

# ------------------- 1️⃣ Load External Dataset -------------------
dataset = load_dataset("imdb")

# 1. Prepare the small training and test sets as before
train_full = dataset['train'].shuffle(seed=42).select(range(500))
test_dataset = dataset['test'].shuffle(seed=42).select(range(100))

# 2. Split the training data into Train (80%) and Validation (20%)
# This creates a 'train' and 'test' key within the train_full object
split_data = train_full.train_test_split(test_size=0.2, seed=42)

train_dataset = split_data['train']
val_dataset = split_data['test'] # This is your new validation set

# ------------------- 2️⃣ Load Pretrained Tokenizer + Model -------------------
model_name = "distilbert-base-uncased"  # small, CPU-friendly model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ------------------- 3️⃣ Tokenize Dataset -------------------
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)


# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# ------------------- 4️⃣ Define Metrics -------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ------------------- 5️⃣ Training Arguments -------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,          # just 1 epoch for demo
    per_device_train_batch_size=8, # How the model learns. (The math). Modified wiight every 8 reviews. Note below
    per_device_eval_batch_size=8, # any number is ok, the only limitation is the memory, all the validation predications will be considered at the same time.
    eval_strategy="epoch",
    logging_steps=10,
    save_strategy="epoch",          # don't save checkpoints for small demo
    load_best_model_at_end=True,      # Automatically reload the best model weights at the end
    metric_for_best_model="accuracy", # Use accuracy to determine which model is "best"
    save_total_limit=2,               # Keep only the best and the most recent checkpoint (saves disk space)
    learning_rate=2e-5,
    report_to="none"             # disable wandb logging
)


# ------------------- 6️⃣ Trainer -------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)] # Stop if accuracy doesn't improve for 1 epoch
)

# ------------------- 7️⃣ Train -------------------
trainer.train()

# ------------------- 8️⃣ Evaluate -------------------
results = trainer.evaluate()
print("Evaluation results:", results)

# Manually evaluate on the unseen test set
final_test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Final Unbiased Test Results:", final_test_results)

import json

# Save the history to a file
with open("training_history.json", "w") as f:
    json.dump(trainer.state.log_history, f)
print("History saved to training_history.json")

# Save Your Fine-Tuned Model
model.save_pretrained("./my_imdb_model")
tokenizer.save_pretrained("./my_imdb_model")

import matplotlib.pyplot as plt
import pandas as pd

# 1. Convert the internal log history to a DataFrame
df = pd.DataFrame(trainer.state.log_history)

# 2. Extract training logs (contain 'loss')
train_logs = df[df['loss'].notna()]

# 3. Extract validation logs (contain 'eval_loss')
# Note: eval_loss is only recorded at the end of each epoch
val_logs = df[df['eval_loss'].notna()]

# 4. Create the dual plot
plt.figure(figsize=(10, 6))

# Plot Training Loss (logged every few steps)
plt.plot(train_logs['epoch'], train_logs['loss'], 
         label='Training Loss', color='blue', marker='o')

# Plot Validation Loss (logged once per epoch)
plt.plot(val_logs['epoch'], val_logs['eval_loss'], 
         label='Validation Loss', color='red', linestyle='--', marker='s')

plt.title('DistilBERT: Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()











# Note:
# The Combined Loss Calculation
# The Cross-Entropy Loss function receives 8 "logits" (the model's raw guesses) and 8 "labels" (the actual answers):
# It calculates how "wrong" the model was for each of the 8 reviews internally.
# It immediately averages those 8 errors into a single number
# The Single "Step"
# Because the loss is now a single average number for that batch, the model:
# Calculates one set of gradients (the direction to change the weights).
# Performs one update to its 66 million parameters.