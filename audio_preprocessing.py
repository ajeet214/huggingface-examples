from datasets import load_dataset, Audio
from transformers import AutoProcessor


dataset = load_dataset("")["train"]

# Resample the audio to a frequency of 16,000 Hz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

# Load the audio processor
processor = AutoProcessor.from_pretrained("openai/whisper-small")

# Preprocess the audio data of the 0th dataset element
audio_pp = processor(dataset[0]['audio']['array'],
                     sampling_rate=16_000, return_tensors='pt')




