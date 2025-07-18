from datasets import load_dataset
from huggingface_hub import HfFolder
import pandas as pd
import os

def process_and_save_hf_data():
    # Retrieve the Hugging Face authentication token
    hf_token = HfFolder.get_token()
    print("Using Hugging Face token:", hf_token)
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please log in using `huggingface-cli login`.")

    # Set the token as an environment variable
    os.environ["HF_AUTH_TOKEN"] = hf_token

    # Load the dataset from Hugging Face
    ds = load_dataset("qualifire/Qualifire-prompt-injection-benchmark")

    # Print available splits for debugging
    print("Available splits:", ds)

    # Convert the dataset to a DataFrame (use the 'test' split)
    df = pd.DataFrame(ds["test"])  # Use the 'test' split

    # Extract relevant fields (e.g., 'text') and add a 'label' column
    if "text" in df.columns:
        df_processed = df[["text", "label"]].copy()
    else:
        raise ValueError("The dataset does not contain the expected columns ('text', 'label').")

    # Rename 'text' column to 'prompt' for consistency
    df_processed.rename(columns={"text": "prompt"}, inplace=True)

    # Ensure the 'preprocessed' folder exists
    output_folder = "data/processed"
    os.makedirs(output_folder, exist_ok=True)

    # Save the processed data to a CSV file
    output_csv_path = os.path.join(output_folder, "qualifire_prompts_labeled.csv")
    df_processed.to_csv(output_csv_path, index=False)

    print(f"Processed data saved to {output_csv_path}")

# Run the processing function
if __name__ == "__main__":
    process_and_save_hf_data()