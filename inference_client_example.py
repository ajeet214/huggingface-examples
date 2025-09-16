"""
Inference providers:

In some cases, you may lack the hardware to run Hugging Face models locally.
Large-parameter LLMs, and image and video generation models in particular often
require Graphics Processing Units (GPUs) to parallelize the computations.
Hugging Face providers inference providers to outsource this hardware to third-party partners.
"""
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    provider="groq",
    api_key=os.getenv("GROQ_API_KEY")
)

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "user",
         "content": "what is the capital of Vietnam?"}
    ]
)

print(completion.choices[0].message.content)
