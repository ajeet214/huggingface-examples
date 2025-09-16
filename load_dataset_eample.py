from datasets import load_dataset

# data = load_dataset("")
data = load_dataset("IVN-RIN/BioBERT_Italian", split="train")

# Filter the dataset for rows with the term "bella" in the text column
filtered = data.filter(lambda row: " bella " in row['text'])
print(filtered)

# select the first 2 rows from the filtered dataset
sliced = filtered.select(range(2))
