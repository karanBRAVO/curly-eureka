import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

prompt = """Extract product details such as weight, voltage, width, height, volume, depth, and wattage from the image. If any detail is not present, indicate 'NA'. Output format: JSON
"""

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(
    conversation, add_generation_prompt=True)

image_file = "https://m.media-amazon.com/images/I/110EibNyclL.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image,
                   text=prompt,
                   return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
