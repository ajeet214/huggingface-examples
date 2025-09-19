from datasets import load_dataset, Audio
from transformers import AutoProcessor, WhisperForConditionalGeneration

dataset = load_dataset("")["train"]

# Resample the audio to a frequency of 16,000 Hz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

sample = dataset[0]['audio']

# Load the pretrained model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids=None

# Load the audio processor
processor = AutoProcessor.from_pretrained("openai/whisper-small")

# Preprocess the sample audio
input_preprocessed = processor(sample['array'],
                               sampling_rate=sample['sampling_rate'],
                               return_tensors='pt',
                               return_attention_mask=True)

# Generate the IDs of the recognized tokens
predicted_ids = model.generate(input_preprocessed.input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)