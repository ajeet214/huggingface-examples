from huggingface_hub import HfApi

api = HfApi()
models = api.list_models(
    task="automatic-speech-recognition",  # Filter by task directly
    author="openai",                      # Optional: filter by author
    library="transformers",               # Optional: filter by library
    limit=10                              # Optional: limit results
)
models = list(models)
print(len(models))
print(models[0].id)
