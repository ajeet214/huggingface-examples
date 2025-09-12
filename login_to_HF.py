import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv("HUGGINGFACE_API_KEY"))

