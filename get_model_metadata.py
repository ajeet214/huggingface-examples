import os
from huggingface_hub import HfApi
from dotenv import load_dotenv
api = HfApi()

load_dotenv()

info = api.model_info("meta-llama/Llama-3.1-8B-Instruct",
                      token=os.getenv("HUGGINGFACE_API_KEY")
                      )

print(f"Author: {info.author}")
print(f"Tags: \n{info.tags}")
print(f"Likes: {info.likes}")
print(f"Inference status: {info.inference}")
print(f"Downloads: {info.downloads}")
print("Inference API available:", info.pipeline_tag is not None)
print("Pipeline type:", info.pipeline_tag)
