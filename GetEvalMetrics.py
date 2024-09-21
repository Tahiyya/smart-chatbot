import pandas as pd
from bert_score import score

# Load your data into a DataFrame
df = pd.read_excel(r"C:\Users\tahiy\VS Code Scripts\Chatbot-Retrieval\Evaluation\Eval_Dataset_Output.xlsx")

# Iterate over the question-answer pairs and calculate BERTScore
precisions, recalls, f1_scores = [], [], []

for idx, row in df.iterrows():
    generated_answer = [row['Actual Output']]  # BERTScore expects lists of strings Actual Output
    reference_answer = [row['Ground Truth']]
    
    # Calculate BERTScore
    P, R, F1 = score(generated_answer, reference_answer, lang="en")
    
    # Append results to lists
    precisions.append(P.mean().item())  # Converting tensor to scalar
    recalls.append(R.mean().item())
    f1_scores.append(F1.mean().item())

# Add results to DataFrame
df['BERTScore Precision'] = precisions
df['BERTScore Recall'] = recalls
df['BERTScore F1'] = f1_scores

# Print the DataFrame with the BERTScore results
# print(df[['Questions', 'BERTScore Precision', 'BERTScore Recall', 'BERTScore F1']])

# Assuming df contains the BERTScore Precision, Recall, and F1 columns

# Calculate the mean for each metric across all rows
final_precision = df['BERTScore Precision'].mean()
final_recall = df['BERTScore Recall'].mean()
final_f1 = df['BERTScore F1'].mean()

# Print the final averaged scores
print(f"Final Precision: {final_precision:.4f}")
print(f"Final Recall: {final_recall:.4f}")
print(f"Final F1 Score: {final_f1:.4f}")

df.to_excel(r"C:\Users\tahiy\VS Code Scripts\Chatbot-Retrieval\Evaluation\Eval_Dataset_Metrics.xlsx")

# Path to save the file
file_path = r'C:\Users\tahiy\VS Code Scripts\Chatbot-Retrieval\Evaluation\Evaluation_Metrics.txt'

# Open the file in write mode and save variables
with open(file_path, 'w') as file:
    file.write(f"Evaluation Metrics for Chatbot.'\n")
    file.write(f"Precision: {final_precision}\n")
    file.write(f"Recall: {final_recall}\n")
    file.write(f"F1 Score: {final_f1}\n")

# Inform the user the file is saved
print(f"Evaluation metric scores saved to {file_path}")
