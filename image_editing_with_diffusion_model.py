"""
## Custom image editing:
AI image generation is already pretty cool, but some models even support custom image editing,
a multi-modal variant of image generation that takes both a text prompt and source image input.
Have a go at modifying this famous self-portrait of Van Gogh to be of the cartoon character Snoopy
using the StableDiffusionControlNetPipeline
"""
import cv2
from PIL import Image
import numpy as np

from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch


image = load_image("http://301.nz/o81bf")

image = cv2.Canny(np.array(image), 100, 200)
image = image[:, :, None]
image.concatenate([image, image, image], axis=2)

# Load a ControlNetModel from the pretrained checkpoint
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

# Load a pretrained StableDiffusionControlNetPipeline using the ControlNetModel
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)

pipe = pipe.to("cuda")

prompt = ["Albert Einstein, bast quality, extremely detailed"]

generator = [torch.Generator(device="cuda").manual_seed(2)]

# Run the pipeline
output = pipe(
    prompt,
    canny_image,
    negative_prompt=[],
    generator=generator,
    num_inference_steps=20
)

