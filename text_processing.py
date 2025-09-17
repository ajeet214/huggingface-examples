from transformers import AutoTokenizer

# Load the tokenizer of the pretrained model
tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

text = ""

# print(tokenizer.backend_tokenizer.normalizer.normalize_str(text))

# Perform full preprocessing of the text
encoded_input = tokenizer(text, retuen_tensors='pt', padding=True)
print(encoded_input)