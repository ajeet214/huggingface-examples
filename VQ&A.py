from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

image = Image.open("D:\\LifeLongLearning\\image.jpg")
text = "What was the gross income in 2011-2012?"

processor = ViltProcessor.from_pretrained("")
model = ViltForQuestionAnswering.from_pretrained("")

# Preprocess the text prompt and image
encoding = processor(image, text, return_tensors="pt")

# Generate the answer tokens
outputs = model(**encoding)

# Find the ID of the answer with the highest confidence
idx = outputs.logits.argmax(-1).item()
print(f"Predicted answer:", model.config.id2label[idx])