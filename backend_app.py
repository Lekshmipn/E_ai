from flask import Flask, request, jsonify
import pandas as pd
import random
import os
from transformers import DebertaV2Tokenizer

app = Flask(__name__)

# Load dataset once when the app starts
DATA_FILE = "train.csv"
ANNOTATED_FILE = "annotations.csv"

# Read the dataset into a Pandas DataFrame
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    raise FileNotFoundError(f"{DATA_FILE} not found.")

# Ensure annotated file exists
if not os.path.exists(ANNOTATED_FILE):
    pd.DataFrame(columns=["issue", "post_text", "annotations"]).to_csv(ANNOTATED_FILE, index=False)

# Initialize DeBERTa-v2 tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")

@app.route('/get-entry', methods=['GET'])
def get_random_entry():
    """
    Fetches a random entry from the train.csv dataset.
    """
    random_entry = df.sample(n=1).iloc[0]
    
    # Tokenize the post_text
    tokens = tokenizer.tokenize(random_entry['post_text'])
    # Convert tokens back to a more readable format
    readable_tokens = [token.replace('▁', '') for token in tokens]
    response = {
        "issue": random_entry['issue'],
        "post_text": random_entry['post_text'],
        "tokens": tokens,  # Add tokenized words
        "readable_tokens":readable_tokens,
        "classification": [
        int(random_entry['Inappropriateness']),
        int(random_entry['Toxic Emotions']),
        int(random_entry['Excessive Intensity']),
        int(random_entry['Emotional Deception']),
        int(random_entry['Missing Commitment']),
        int(random_entry['Missing Seriousness']),
        int(random_entry['Missing Openness']),
        int(random_entry['Missing Intelligibility']),
        int(random_entry['Unclear Meaning']),
        int(random_entry['Missing Relevance']),
        int(random_entry['Confusing Reasoning']),
        int(random_entry['Other Reasons']),
        int(random_entry['Detrimental Orthography']),
        int(random_entry['Reason Unclassified']),
                            ]
    }
    return jsonify(response)
    #print("response is ", response)

# Global variables
current_index = None
balanced_sample = None

# To implement balanced post entries
def get_balanced_sample(df, n=30):
    """
    Fetches n entries from the dataset with a balanced distribution of labels.
    """
    global balanced_sample

    # Define the columns with the labels
    label_columns = [
        "Inappropriateness", "Toxic Emotions", "Excessive Intensity", 
        "Emotional Deception", "Missing Commitment", "Missing Seriousness", 
        "Missing Openness", "Missing Intelligibility", "Unclear Meaning", 
        "Missing Relevance", "Confusing Reasoning", "Other Reasons", 
        "Detrimental Orthography", "Reason Unclassified"
    ]
    
    # Create an empty DataFrame for the balanced sample
    balanced_sample = pd.DataFrame()

    # Calculate number of samples per label for balance
    samples_per_label = n // len(label_columns)

    for label in label_columns:
        # Filter rows where the current label is 1
        positive_entries = df[df[label] == 1]
        if len(positive_entries) < samples_per_label:
            raise ValueError(f"Not enough entries for label '{label}' to ensure balance.")
        
        # Select the first 'samples_per_label number' of entries for the label
        sampled_entries = positive_entries.head(samples_per_label)
        balanced_sample = pd.concat([balanced_sample, sampled_entries], ignore_index=True)

    # Save the balanced dataset to a CSV file
    #balanced_sample.to_csv("balanced_entries_42.csv", index=False)
    # Shuffle the final balanced sample to ensure randomness across labels
    # Remove the shuffle step if order consistency is preferred
    #balanced_sample = balanced_sample.sample(frac=1).reset_index(drop=True)

    return balanced_sample



@app.route('/get-balanced-entry', methods=['GET'])
def get_balanced_entry():
    """
    Returns the next entry from the balanced dataset, keeping track of the current index.
    """
    global current_index, balanced_sample  # Declare globals to modify them

    # Initialize the balanced dataset if not already created
    if balanced_sample is None or current_index is None:
        balanced_sample = get_balanced_sample(df, n=28)  # Adjust n as needed, better if multiples of 14
        current_index = 0  # Start from the beginning

    # Get the current entry
    entry = balanced_sample.iloc[current_index]

    # Prepare the response
    tokens = tokenizer.tokenize(entry['post_text'])
    readable_tokens = [token.replace('▁', '') for token in tokens]

    response = {
        "issue": entry['issue'],
        "post_text": entry['post_text'],
        "tokens": tokens,
        "readable_tokens": readable_tokens,
        "classification": [
            int(entry['Inappropriateness']),
            int(entry['Toxic Emotions']),
            int(entry['Excessive Intensity']),
            int(entry['Emotional Deception']),
            int(entry['Missing Commitment']),
            int(entry['Missing Seriousness']),
            int(entry['Missing Openness']),
            int(entry['Missing Intelligibility']),
            int(entry['Unclear Meaning']),
            int(entry['Missing Relevance']),
            int(entry['Confusing Reasoning']),
            int(entry['Other Reasons']),
            int(entry['Detrimental Orthography']),
            int(entry['Reason Unclassified']),
        ]
    }

    # Update the index for the next call
    current_index += 1

    # Reset the index if we reach the end of the dataset
    if current_index >= len(balanced_sample):
        current_index = 0  # Loop back to the beginning

    return jsonify(response)

@app.route('/save-annotation', methods=['POST'])
def save_annotation():
    """
    Saves user annotations to a separate CSV file.
    Expects JSON with 'issue', 'post_text', and 'annotations' keys.
    """
    data = request.json
    if not all(k in data for k in ("issue", "post_text", "annotations")):
        return jsonify({"error": "Invalid data format"}), 400
    
    # Append to annotation file
    new_entry = pd.DataFrame([data])
    new_entry.to_csv(ANNOTATED_FILE, mode='a', header=not os.path.exists(ANNOTATED_FILE), index=False)
    
    return jsonify({"message": "Annotation saved successfully!"}), 200

# implement distributed dataset - 30 
# send from this datset to frond end and cross check with tracker

if __name__ == '__main__':
    app.run(debug=True, port=5001)

