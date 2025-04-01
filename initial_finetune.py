import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import TrainerCallback
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import evaluate



# Load Tokenizers and Models
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)
    return tokenizer, model

# Define Models
models = {
    "bert-base-uncased": load_model_and_tokenizer("bert-base-uncased"),
   "roberta-base": load_model_and_tokenizer("roberta-base")
   # "distilbert-base-uncased": load_model_and_tokenizer("distilbert-base-uncased")
}


#yelp_dataset = load_dataset("yelp_review_full")
sst_dataset = load_dataset("sst2")
imdb_dataset = load_dataset("imdb")

sst_dataset = sst_dataset['train'].train_test_split(test_size=0.2, seed=42)
imdb_dataset = imdb_dataset['train'].train_test_split(test_size=0.2, seed=42)

sst_train = sst_dataset['train']
sst_test = sst_dataset['test']

imdb_train = imdb_dataset['train']
imdb_test = imdb_dataset['test']

datasets = {
    'sst': (sst_train, sst_test),
    'imdb': (imdb_train, imdb_test)
}


# Load accuracy metric from Hugging Face evaluate
accuracy_metric = evaluate.load("accuracy")

class MetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(f"Epoch {state.epoch}: Loss = {metrics['eval_loss']}, Accuracy = {metrics.get('eval_accuracy', 'N/A')}")



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)






#finetune 12 models
# Tokenize Data
def preprocess_function(examples, tokenizer):
    # Detect input column name
    text_column = 'text' if 'text' in examples else 'sentence'
    return tokenizer(examples[text_column], padding='max_length', truncation=True)


def fine_tune_model(model_name, tokenizer, model, train_dataset, test_dataset, num_epochs):
    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=f"./{model_name}_{dataset_name}_finetuned",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=f"./logs/{model_name}_{dataset_name}",
        save_strategy="epoch",
        fp16=True,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MetricsCallback()]
    )

    print(f"Starting fine-tuning for {model_name} on {dataset_name} for {num_epochs} epochs...")
    trainer.train()

    # Save the Model and Tokenizer
    save_path = f"./{model_name}_{dataset_name}_finetuned"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


epochs_per_model = {
"bert-base-uncased": 3,
"roberta-base": 4
}
for model_name, (tokenizer, model) in models.items():
    for dataset_name, (train_dataset, test_dataset) in datasets.items():
        num_epochs = epochs_per_model.get(model_name, 3)
        print(f"\n--- Fine-tuning {model_name} on {dataset_name} for {num_epochs} epochs ---")
        fine_tune_model(f"{model_name}_{dataset_name}", tokenizer, model, train_dataset, test_dataset, num_epochs)

print("All models have been fine-tuned and saved.")
