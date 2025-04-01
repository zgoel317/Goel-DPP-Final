import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import pandas as pd
import stanza
import nltk
from nltk import Tree
import matplotlib


# Initialize the Stanza pipeline for constituency parsing
# Initialize Stanza with GPU enabled
nlp = stanza.Pipeline('en', processors='tokenize,pos,constituency', use_gpu=True)


sst_dataset = load_dataset("sst2")
#yelp_dataset = load_dataset("yelp_review_full")
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

def generate_parse_tree(sentence):
    """
    Generate a parse tree for a given sentence using Stanza.
    """
    doc = nlp(sentence)
    for sent in doc.sentences:
        parse_tree = sent.constituency
        return str(parse_tree)

# Apply the parser to generate parse trees for both datasets
sst_test_df = pd.DataFrame(sst_test)
imdb_test_df = pd.DataFrame(imdb_test)

sst_test['parse_tree'] = sst_test['sentence'].apply(generate_parse_tree)
imdb_test['parse_tree'] = imdb_test['text'].apply(generate_parse_tree)

# Save to CSV for future use
sst_test.to_csv("sst_test_with_parsetrees.csv", index=False)
imdb_test.to_csv("imdb_test_with_parsetrees.csv", index=False)

sst_train_df = pd.DataFrame(sst_train)
imdb_train_df = pd.DataFrame(imdb_train)

sst_train_df['parse_tree'] = sst_train_df['sentence'].apply(generate_parse_tree)
imdb_train_df['parse_tree'] = imdb_train_df['text'].apply(generate_parse_tree)

# Save to CSV for future use
sst_train_df.to_csv("sst_test_with_parsetrees.csv", index=False)
imdb_train_df.to_csv("imdb_test_with_parsetrees.csv", index=False)



print("Parse trees generated and saved!")


def visualize_parse_tree(parse_tree_str):
    """
    Visualize a parse tree using NLTK.
    """
    try:
        tree = Tree.fromstring(parse_tree_str)
        tree.pretty_print()
        tree.draw()
    except Exception as e:
        print("Error visualizing tree:", e)

# Example usage for one sentence from SST-2 and IMDB
print("SST-2 Example Parse Tree:")
visualize_parse_tree(sst_test['parse_tree'].iloc[0])

print("IMDB Example Parse Tree:")
visualize_parse_tree(imdb_test['parse_tree'].iloc[0])