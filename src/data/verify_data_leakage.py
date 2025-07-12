import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from difflib import SequenceMatcher
import numpy as np

# Load train and test data
train_df = pd.read_csv("./data/preprocessed/train_data.csv")
test_df = pd.read_csv("./data/preprocessed/test_data.csv")

print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

# 1️⃣ Exact prompt overlap
exact_overlap = set(train_df["prompt"]).intersection(set(test_df["prompt"]))
print(f"🔁 Exact Overlaps: {len(exact_overlap)}")

# 2️⃣ Fuzzy text similarity (optional: slow on large sets, sample for speed)
sampled_train = train_df["prompt"].sample(n=500, random_state=42).tolist()
sampled_test = test_df["prompt"].sample(n=500, random_state=42).tolist()

similar_pairs = []
for t in tqdm(sampled_test, desc="🔍 Checking fuzzy matches"):
    for tr in sampled_train:
        score = SequenceMatcher(None, t, tr).ratio()
        if score > 0.9:
            similar_pairs.append((t, tr, score))

print(f"🤖 Fuzzy similar pairs > 0.9: {len(similar_pairs)}")

# 3️⃣ N-gram frequency bias check
combined_df = pd.concat([train_df, test_df], ignore_index=True)
label_map = {label: i for i, label in enumerate(combined_df["label"].unique())}
combined_df["label_num"] = combined_df["label"].map(label_map)

vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words="english", max_features=5000)
X = vectorizer.fit_transform(combined_df["prompt"])
X_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
X_df["label"] = combined_df["label"]

# Mean frequency of n-grams per label
means = X_df.groupby("label").mean()
diff = (means.loc["adversarial"] - means.loc["benign"]).abs().sort_values(ascending=False)

print("\n🔎 Top potential leakage n-grams:")
print(diff.head(20))
