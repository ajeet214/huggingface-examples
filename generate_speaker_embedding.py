import torch
from datasets import load_dataset
from speechbrain.inference.speaker import EncoderClassifier

dataset = load_dataset("")
speaker_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")

speaker_embeddings = speaker_model.encode_batch(torch.tensor(dataset[0]["audio"]["array"]))
speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2).unsqueeze(0)