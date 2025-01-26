#code to process the gpt prediction and convert it into a format providing token-level insights into how much of the actual text are present in the predicted tokens.
import pandas as pd
import ast
import re

# Load the original dataset and the predictions data
original_data = pd.read_csv('/Dataset/balanced_test_entries_42.csv')
predictions_data = pd.read_csv('results.txt')

# Clean up the 'labels' column by replacing invalid quote characters and ensuring proper formatting
def clean_labels_string(labels_string):
    # Ensure the dictionary keys and strings are properly quoted
    # Replace curly quotes and other non-standard characters with standard quotes
    labels_string = labels_string.replace("‘", "'").replace("’", "'").replace('“', '"').replace('”', '"')
    
    # Ensure that the keys are enclosed in quotes (i.e., make them valid string keys)
    labels_string = re.sub(r'(\w+)(:)', r'"\1"\2', labels_string)  # Add quotes around keys
    
    # Return the cleaned-up string
    return labels_string

# Apply the cleaning function to the 'labels' column
predictions_data['labels'] = predictions_data['labels'].apply(clean_labels_string)

# Create a new list to store the rows for the new CSV
new_rows = []

# Iterate through each row in the predictions data
for index, pred_row in predictions_data.iterrows():
    post_id = pred_row['post_id']
    method_used = pred_row['method_used']
    
    # Extract the post_text for this post_id
    post_text_row = original_data[original_data['post_id'] == post_id]
    if not post_text_row.empty:
        post_text = post_text_row.iloc[0]['post_text']
    else:
        continue
    
    # Tokenize the post_text (split by spaces)
    post_tokens = set(post_text.lower().split())  # Using a set for faster lookup

    # Parse the label dictionary from the predictions using ast.literal_eval
    try:
        labels_dict = ast.literal_eval(pred_row['labels'])
    except Exception as e:
        print(f"Error parsing labels for post_id {post_id}: {e}")
        continue  # Skip this row if it cannot be parsed

    # Construct the new labels
    new_labels = {}
    for label, tokens in labels_dict.items():
        # For each label, check if the token is present in post_tokens
        label_tokens_with_status = [(token, 1 if token in post_tokens else 0) for token in tokens]
        new_labels[label] = label_tokens_with_status
    
    # Add the result for this row to the new_rows list
    new_row = {
        'post_id': post_id,
        'method_used': method_used,
        'labels': str(new_labels)  # Convert the new labels dictionary to a string
    }
    new_rows.append(new_row)

# Convert the list of new rows to a DataFrame
new_data = pd.DataFrame(new_rows)

# Save the new data to a CSV file
new_data.to_csv('new_labels_output.csv', index=False)
