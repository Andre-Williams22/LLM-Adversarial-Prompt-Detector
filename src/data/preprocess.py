import pandas as pd
from sklearn.model_selection import train_test_split

# Read in the cleaned dataset
data = pd.read_csv('/Users/andrewilliams/Documents/development/projects/LLM-Adversarial-Prompt-Detector/src/data/wildjailbreak_cleaned.csv')

# Drop duplicates and nulls
data = data.drop_duplicates().dropna()

# Map labels
data['label'] = data['label'].map({'adversarial': 1, 'safe': 0})

# Split the data into features and target
X = data.drop('label', axis=1)
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the splits to data/processed/
X_train.to_csv('/Users/andrewilliams/Documents/development/projects/LLM-Adversarial-Prompt-Detector/src/data/processed/X_train.csv', index=False)
X_test.to_csv('/Users/andrewilliams/Documents/development/projects/LLM-Adversarial-Prompt-Detector/src/data/processed/X_test.csv', index=False)
y_train.to_csv('/Users/andrewilliams/Documents/development/projects/LLM-Adversarial-Prompt-Detector/src/data/processed/y_train.csv', index=False)
y_test.to_csv('/Users/andrewilliams/Documents/development/projects/LLM-Adversarial-Prompt-Detector/src/data/processed/y_test.csv', index=False)