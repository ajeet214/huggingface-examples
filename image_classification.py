from datasets import load_dataset
from transformers import pipeline

# load the 134th image data
dataset = load_dataset("")
image = dataset['test'][134]['image']

# Load the image-classification pipeline
pipe = pipeline(
    task="image-classification",
    model="google/mobilenet_v2_1.0_224"
)

# Use the pipeline to predict the class of the sample image
pred = pipe(image)

# Print the first (highest probability) label
print(f"predicted class: {pred[0]['label']}")
