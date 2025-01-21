import numpy as np
import pandas as pd
import csv  # For saving results to a CSV file
import lime
import lime.lime_text
import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pickle
from IPython.display import FileLink

# Define the inappropriateness dimensions and labels
DIMS = [
    'Inappropriateness', 'Toxic Emotions', 'Excessive Intensity', 'Emotional Deception', 
    'Missing Commitment', 'Missing Seriousness', 'Missing Openness', 'Missing Intelligibility', 
    'Unclear Meaning', 'Missing Relevance', 'Confusing Reasoning', 'Other Reasons', 
    'Detrimental Orthography', 'Reason Unclassified'
]


# Adjust the model path to point to the Kaggle dataset
model_path = '/kaggle/input/distilbert'  

# Load the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=len(DIMS))

# Define the prediction function for LIME
def predict_fn(texts):
    # Tokenize the input texts
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    # Make predictions using the model
    with torch.no_grad():
        logits = model(**inputs).logits
    # Apply sigmoid to logits to get probabilities
    probs = torch.sigmoid(logits).numpy()
    return probs


# Define the function to process the label explanations
def process_labels(post_text, label_explanations):
    # Tokenize the post text
    #post_words = set(post_text.lower().split())  # Convert to lower case for case-insensitivity

    # Tokenize the post text while maintaining the original order
    post_words = post_text.split()  # Keep words in their original order
    #print(f"Post Words: {post_words}\n")
    
    # Create a dictionary to store the new label entries
    processed_labels = {}

    # Iterate through the existing label explanations
    for label, annotations in label_explanations:
        # Create a new list to store the word-value pairs
        new_annotations = []

        # Iterate through each word in the post text
        for word in post_words:  # Now iterating through words in the post_text
            # Check if the word is present in the annotations
            #if word in [annotation_word.lower() for annotation_word in annotations]:
            # Check if the word is present in the annotations
            if word.lower() in [annotation_word.lower() for annotation_word in annotations]:
                new_annotations.append((word, 1))  # Word exists in annotations, so value is 1
            else:
                new_annotations.append((word, 0))  # Word doesn't exist in annotations, so value is 0

        # Add the processed label to the dictionary
        processed_labels[label] = new_annotations
        #print("processed labels are...",processed_labels[label])

    return processed_labels


# Create the explainer
explainer = lime.lime_text.LimeTextExplainer(class_names=DIMS)


# Read the CSV file 
csv_file_path = '/kaggle/input/testdata/balanced_test_entries.csv' 
data = pd.read_csv(csv_file_path)

# Prepare an empty list to store the results
results = []

# Process each row in the CSV
for idx, row in data.iterrows():
    post_id = row['post_id']  # Extract post ID
    post_text = row['post_text']  # Extract the text to be analyzed
    print(f"\nAnalyzing post_id: {post_id}")
    print(f"Post Text: {post_text}\n")


    # Initialize a dictionary to store the result for this post
    result = {"post_id": post_id, "method_used": "LIME"}

    # Get predicted probabilities
    prob_pr = predict_fn([post_text])  # Note: Pass text as a list
    print("Predicted probabilities (labels with prob >= 0.5):")
    
    # Create a list to store the label explanations
    label_explanations = []
    
    # Explain the prediction for labels with prob >= 0.5
    exp = explainer.explain_instance(
        post_text, 
        predict_fn, 
        labels=list(range(len(DIMS)))  # All label indices
    )

    for i, label in enumerate(DIMS):
        if prob_pr[0][i] >= 0.5:  # Only process labels with prob >= 0.5
            #print(f"  {label}: {prob_pr[0][i]:.4f}")
            # Get the influential words for this label
            influential_words = exp.as_list(label=i)  # List of (word, contribution) pairs
            annotations = [word for word, contribution in influential_words]  # Extract tokens only
            
            # Format the explanation as a tuple (label, [tokens])
            label_explanations.append((f"ann_{label.lower().replace(' ', '_')}", annotations))
    
    # Process the labels to match the required format
    processed_labels = process_labels(post_text, label_explanations)

    # Store the processed labels in the result dictionary
    result["labels"] = str(processed_labels)

    # Append the result to the results list
    results.append(result)

    print("-" * 50)  # Separator for readability

# Save results to a CSV file
output_csv_file = 'lime_results_test.csv'
with open(output_csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
    
# Save results to a Pickle file
output_pickle_file = '/kaggle/working/lime_results_test.pkl'  # Specify the Pickle file path
with open(output_pickle_file, 'wb') as file:
    pickle.dump(results, file)  # Serialize and save the results object

# Save the explanation as HTML
#exp.save_to_file('lime_explanation_test.html')

# Save the file to the Kaggle working directory
output_file_path = '/kaggle/working/lime_results_test.csv'
# Create a direct link to the file
FileLink(output_file_path)

print("Download your results:")
FileLink(output_pickle_file)
