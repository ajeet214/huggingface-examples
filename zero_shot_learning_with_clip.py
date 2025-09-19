from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset

dataset = load_dataset("")

model = CLIPModel.from_pretrained("")
processor = CLIPProcessor.from_pretrained("")

categories = ["shirt", "trousers", "shoes", "dress", "hat",
              "bag", "watch", "glasses", "jacket", "belt"]

# Preprocess the categories and image
inputs = processor(
    text=categories,
    images=dataset["train"][0]["image"],
    return_tensors="pt",
    padding=True
)

# Process the unpacked inputs with the model
outputs = model(**inputs)

# Calculate the probabilities of each category
probs = outputs.logits_per_image.softmax(dim=1)

# Find the most likely category
category = categories[probs.argmax().item()]
print(f"Predicted category: {category}")
