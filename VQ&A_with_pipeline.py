"""
LayoutLM: Trained with images with Q/As from the DocVQA dataset

"""
from transformers import pipeline
from datasets import load_dataset


dataset = load_dataset("lmms-lab/DocVQA")

# Load the document-question-answering pipeline with the pretrained model
pipe = pipeline(task="document-question-answering", model="impira/layoutlm-document-qa")

# Process datapoint 61 to find the amount of training days
result = pipe(dataset["test"][61]["image"], "how many days of formal training were provided to employees in 2012-2013?")


print(result)
# [{'score': 0.9520635604858398, 'answer': '1,22,000', 'start': 9, 'end': 9}]