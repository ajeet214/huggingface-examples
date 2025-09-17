from datasets import load_dataset
from transformers import BlipProcessor, Blip2ForConditionalGeneration

checkpoint = "Salesforce/blip-image-captioning-base"
model = Blip2ForConditionalGeneration.from_pretrained(checkpoint)

# Load the image processor of the pretrained model
processor = BlipProcessor.from_pretrained(checkpoint)

# Load the image from index 11 of the dataset
image = load_dataset("")['test'][11]['image']

# Preprocess the image
inputs = processor(images=image, return_tensors='pt')

# Generate a caption using the model
output = model.generate(**inputs)

print(f'Generated caption: {processor.decode(output[0])}')

sample = load_dataset("")['test'][11]
print(f'Original caption: {sample["caption"][0]}')
