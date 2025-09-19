import numpy as np
import torch
from diffusers import CogVideoXPipeline

prompt = "A robot doing the robot dance. The dance floor has colorful squares and a glitterball."


# Create a CogVideoXPipeline
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16
)

pipe.enable_model_cpu_offload()
pipe.enalbe_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()


# Run the pipeline with the provided prompt
video = pipe(
    prompt=prompt,
    num_inference_steps=20,
    num_frames=20,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42)
).frames[0]


from diffusers.utils import export_to_video
from moviepy.editor import VideoFileClip

video_path = VideoFileClip(video_path)
video.write_gif("video.gif")


from diffusers.utils import load_video
from torchmetices.functional.multimodal import measure_clip_score
from functools import partial
frames = load_video(video_path)


# Setup CLIP scoring
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch32")

frame_tensors = []
for frame in frames:
    frame = np.array(frame)
    frame_int = (frame * 255).astype("uint8")
    frame_tensor = torch.from_numpy(frame_int).permute(2, 0, 1)
    frame_tensors.append(frame_tensor)

# Pass a list of CHW tensors as expected by clip_score
scores = clip_score_fn(frame_tensors, [prompt] * len(frame_tensors)).detach().cpu().numpy()

avg_clip_score = round(np.mean(scores), 4)
print(f"Average CLIP score: {avg_clip_score}")