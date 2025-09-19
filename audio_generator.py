"""
three components to generate audio
- preprocessor: resampling and feature extraction
- model: feature transformation
- vocoder: a separate generative model for audio waveforms
"""
import torch
from datasets import load_dataset
from speechbrain.inference.speaker import EncoderClassifier
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

dataset = load_dataset("")
speaker_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")

speaker_embeddings = speaker_model.encode_batch(torch.tensor(dataset[0]["audio"]["array"]))
speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2).unsqueeze(0)

inputs = processor(
    audio=dataset[0]["audio"],
    sampling_rate=dataset[0]["audio"]["sampling_rate"],
    return_tensors="pt"
)

# Generate the denoised speech
speech = model.generate_speech(inputs['input_values'], speaker_embeddings, vocoder=vocoder)

# make_spectrogram(speech)
# sf.write("speech.wav", speech.numpy(), samplerate=16000)