from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import pandas as pd
import os

dataset_dir_path = "./dataset"
test_file_name = "test.csv"
test_img_dir_path = "./images/test"
test_dataset = os.path.normpath(os.path.join(dataset_dir_path, test_file_name))


def get_image_name_from_url(url: str) -> str:
    return url.split('/')[-1]


model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
)

df = pd.read_csv(test_dataset)
print(df)
test_images = df['image_link'].to_list()
total_images = 2

# prompt = """
# Analyze this image and extract the following specifications. Provide the output in a JSON-like format with key-value pairs. Use the exact keys specified below. Include the unit in the value string. If a specification is not visible or applicable, use 'N/A' as the value.

# {
#     weight:
#     height:
#     length:
#     width:
#     depth:
#     volume:
#     voltage:
#     wattage:
# }

# Only include specifications directly observable in the image. give output in JSON-like structure.
# """

prompt = """
Please analyze the image and provide a detailed list of all measurable properties and specifications shown, including but not limited to:

{
length
width
height
depth
weight
volume
wattage
maximum_weight_recommendation
voltage
}
Any numerical values or measurements displayed
Any text or labels visible

Please list each property with its corresponding value and unit of measurement where applicable. If any standard properties like voltage or wattage are not depicted, simply note that they are not shown in the image. Provide the output in a JSON-like format with key-value pairs. Use the exact keys specified below. Include the unit in the value string. If a specification is not visible or applicable, use 'N/A' as the value.
"""

for i in range(total_images):
    img_name = get_image_name_from_url(test_images[i])
    img_path = os.path.normpath(os.path.join(test_img_dir_path, img_name))

    if not os.path.exists(img_path):
        print(f"[!] {img_path} Not found")
    else:
        print(img_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(output_text)
