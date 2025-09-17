"""
## When to use AutoModels and AutoTokenizers?

Pipelines and AutoModels with AutoTokenizers are two approaches to working with Hugging Face models,
each suited for different use cases. Pipelines offer simplicity, while AutoModels and AutoTokenizers
provide more control and customization.

"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# download the model and tokenizer
my_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-englis"
)
my_tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-englis"
)

# # Split input text into tokens
# tokens = my_tokenizer.tokenize("AI: Making robots smarter and humans lazier!")
#
# # Display the tokenized output
# print(f"Tokenized output: {tokens}")

# create the custom pipeline
my_pipeline = pipeline(task="sentiment-analysis", model=my_model, tokenizer=my_tokenizer)

# Predict the sentiment
output = my_pipeline("This course is pretty good, I guess.")
print(f"Sentiment using AutoClasses: {output[0]['label']}")