from huggingface_hub import snapshot_download

# Downloads model weights/config to ~/.cache/huggingface/hub
local_dir = snapshot_download(repo_id="openai/gpt-oss-20b")

print(f"Model downloaded to: {local_dir}")
