import evaluate
from evaluate import evaluator
from transformers import pipeline
from datasets import load_dataset

# Load a text-to-audio pipeline
pipe = pipeline(task="text-to-audio",
                model="facebook/musicgen-small",
                framework='pt'
                )

# Make a dictionary to set the generation temperature to 0.8 and max_new_tokens to 1
generate_kwargs = {"temperature": 0.8, "max_new_tokens": 1}

# Generate an audio array passing the arguments
outputs = pipe("Classic rock riff", generate_kwargs=generate_kwargs)

# sf.write("output.wav", outputs["audio"][0][0], outputs["sampling_rate"])

# Instantiate the task evaluator
task_evaluator = evaluator("image-classification")
task_evaluator.METRIC_KWARGS = {"average": "weighted"}

metrics_dict = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

# Get label map from pipeline
label_map = pipe.model.config.label2id

dataset = load_dataset("")

# Compute the metrics
eval_results = task_evaluator.compute(
    model_or_pipeline=pipe,
    data=dataset,
    metric=evaluate.combine(metrics_dict),
    label_mapping=label_map
)
print(f"Precision: {eval_results['precision']:.2f}, Recall: {eval_results['recall']:.2f}")