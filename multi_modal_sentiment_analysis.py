from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

dataset = load_dataset("RealTimeData/bbc_news_alltime", "2017-01", split="train")

article_image = dataset[87]["top_image"]
article_text = dataset[87]["content"]


vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "", device_map="auto", torch_dtype="auto"
)

min_pixels = 224 * 224
max_pixels = 448 * 448
vl_model_processor = Qwen2VLProcessor.from_pretrained(
    "", min_pixels=min_pixels, max_pixels=max_pixels
)

text_query = f"Is the sentiment of the following content is good or bad for the Ford share price: {article_text}. Provide reasoning"

# Add the text query dictionary to the chat template
chat_template = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": article_image},
            {"type": "text", "text": text_query}
        ]
    }
]

text = vl_model_processor.apply_chat_template(chat_template=chat_template, tokenize=False, add_generation_prompt=True)

image_inputs, _ = process_vision_info(chat_template)

# Use the processor to preprocess the text and image
inputs = vl_model_processor(text=[text], images=image_inputs, padding=True, return_tensors=True)

# Use the model to generate the output IDs
generated_ids = vl_model.generate(**inputs, max_new_tokens=500)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

# Decode the generated IDs
output_text = vl_model_processor.batch_decode(
       generated_ids_trimmed, skip_special_tokens=True, clea_up_tokenization_spaces=False
)
print(output_text[0])