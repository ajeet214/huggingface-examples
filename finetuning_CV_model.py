from datasets import load_dataset
from transformers import AutoModelForImageClassification
from transformers import AutoImageProcessor
from torchvision.transforms import Compose, Normalize, ToTensor
import matplotlib.pyplot as plt
from transformers import TrainingArguments
from transformers import Trainer, DefaultDataCollator


dataset = load_dataset("")['train']

checkpoint = "google/mobilenet_v2_1.0_224"

# ------------------------------Data preparation-------------------------
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
transform = Compose([ToTensor(), normalize])


def transforms(examples):
    examples['pixel_values'] = [transform(img.convert)("RGB") for img in examples["image"]]
    del examples['image']
    return examples


# Apply the transformations
dataset = dataset.with_transform(transforms)

# Plot the transformed image
plt.imshow(dataset['train'][0]['pixel_values'].permute(1, 2, 0))
plt.show()

# ---------------------model classes-----------------------------
# Create a train/test split within the HF dataset
data_splits = dataset.train_test_split(test_size=0.2, seed=42)

# Obtain the new label names from the dataset
labels = data_splits['train'].features['label'].names

label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label


model = AutoModelForImageClassification.from_pretrained(
    model=checkpoint,
    num_labels=len(labels),
    # Add the id2label mapping
    id2label=id2label,
    # Add the corresponding label2id mapping
    label2id=label2id,
    # Add the required flag to change the number of classes
    ignore_mismatched_sizes=True
)


# -----------------------trainer configuration----------------------------
# training process
training_args = TrainingArguments(
    output_dir="dataset_finetune",
    # Adjust the learning rate
    learning_rate=6e-5,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    push_to_hub=False
)

data_collator = DefaultDataCollator()

trainer = Trainer(
    # Provide the model and datasets
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=image_processor,
    data_collator=data_collator
)
trainer.train()

# evaluation
predictions = trainer.predict(dataset['test'])
print(predictions.metrics['test_accuracy'])
