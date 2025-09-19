import numpy as np
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained(
    "",
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True
)


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32)/ 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32)/ 255.0

    image[image_mask > 0.5] = -1.0
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


control_image = make_inpaint_condition(init_image, mask_image)

# Run the pipeline requesting a black beard
output = pipe(
    "generate a black beard",
    num_inference_steps=40,
    eta=1.0,
    image=init_image,
    mask_image=mask_image,
    control_image=control_image
).images[0]

# plt.imshow(output.images[0])
# plt.show()
