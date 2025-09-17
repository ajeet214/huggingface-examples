from datasets import load_dataset
from transformers import pipeline

# load the 134th image data
dataset = load_dataset("")
image = dataset['test'][134]['image']

# Load the object-detection pipeline
pipe = pipeline(
    task="object-detection",
    model="facebook/detr-resnet-50",
    revision="no_timm")

# prediction
outputs = pipe(image, threshold=0.95)

for obj in outputs:
    box = obj['box']
    print(f"Dectected {obj['label']} with confidence {obj['score']:.2f} at ({box['xmin']}, {box['ymin']}) to ({box['xmax']}, {box['ymax']})")

