#Python Example: Fine-Tuning DistilBERT on IMDB Reviews
# python3 -m venv .venv7 
# source .venv7/bin/activate
# python3 -m venv .vfinetuning
# source .vfinetuning/bin/activate
# python3 Fine-Tuning.py

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

# ------------------- 1️⃣ Load External Dataset -------------------
dataset = load_dataset("imdb")

# For a small demo, we only use 500 training and 100 test samples
train_dataset = dataset['train'].shuffle(seed=42).select(range(500))
test_dataset = dataset['test'].shuffle(seed=42).select(range(100))

# ------------------- 2️⃣ Load Pretrained Tokenizer + Model -------------------
model_name = "distilbert-base-uncased"  # small, CPU-friendly model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ------------------- 3️⃣ Tokenize Dataset -------------------
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
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
    num_train_epochs=1,          # just 1 epoch for demo
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    logging_steps=10,
    save_strategy="no",          # don't save checkpoints for small demo
    learning_rate=2e-5,
    report_to="none"             # disable wandb logging
)

# ------------------- 6️⃣ Trainer -------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# ------------------- 7️⃣ Train -------------------
trainer.train()

# ------------------- 8️⃣ Evaluate -------------------
results = trainer.evaluate()
print("Evaluation results:", results)