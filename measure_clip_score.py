"""
- similarity between encoded image and encoded description
- ranges from 100(perfect agreement) to 0(no agreement)
"""
from datasets import load_dataset
from torchmetrics.functional.multimodal import clip_score
from torchvision.transforms import ToTensor


dataset = load_dataset("")

image = dataset["train"][0]["image"]
description = dataset["train"][0]["Description"]


image = ToTensor()(image)*255

score = clip_score(image, description, "openai/clip-vit-base-patch32")
print(f"Clip score: {score}")
