from datasets import load_dataset
from transformers import pipeline

# load the 134th image data
dataset = load_dataset("")
image = dataset['test'][134]['image']

# Load the image-segmentation pipeline
pipe = pipeline("image-segmentation",
                model="briaai/RMBG-1.4",
                trust_remote_code=True
                )

# Use the pipeline to predict the class of the sample image
outputs = pipe(image)
