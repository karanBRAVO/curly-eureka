from PIL import Image
import requests
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
import pandas as pd
import os
import json
import re

entity_unit_map = {
    'width': {
        'centimetre',
        'foot',
        'inch',
        'metre',
        'millimetre',
        'yard'
    },
    'depth': {
        'centimetre',
        'foot',
        'inch',
        'metre',
        'millimetre',
        'yard'
    },
    'height': {
        'centimetre',
        'foot',
        'inch',
        'metre',
        'millimetre',
        'yard'
    },
    'item_weight': {
        'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'
    },
    'maximum_weight_recommendation': {
        'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'
    },
    'voltage': {
        'kilovolt',
        'millivolt',
        'volt'
    },
    'wattage': {
        'kilowatt',
        'watt'
    },
    'item_volume': {
        'centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart'
    }
}

allowed_units = {
    unit for entity in entity_unit_map for unit in entity_unit_map[entity]}

dataset_dir_path = "./dataset"
test_file_name = "test.csv"
test_img_dir_path = "./images/test"
test_dataset = os.path.normpath(os.path.join(dataset_dir_path, test_file_name))
output_csv_path = os.path.normpath(
    os.path.join(dataset_dir_path, "output.csv"))

model_id = "microsoft/Phi-3-vision-128k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='eager'
)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

df = pd.read_csv(test_dataset)
print(df)
test_images = df['image_link'].to_list()
total_images = 2

messages = [
    {
        "role": "user",
        "content": "<|image_1|>\nWhat is shown in this image?"
    },
    {
        "role": "assistant",
        "content": "The image contains the product along with it's details/specifications."
    },
    {
        "role": "user",
        "content": f"""If image contains the specifications then provide me the specifications along with their units about the product such as it's width, height, depth, volume, voltage, weight, wattage.
        Use the following units:
        - Width, depth, height: metre
        - Weight: kilogram
        - Volume: litre
        - Voltage: volt
        - Wattage: watt
        Provide the response in JSON format. if there are no values then omit those fields."""
    }
]

prompt = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


def normalize_unit(unit):
    """Normalize common abbreviations to their standard unit names."""
    unit_mapping = {
        'cm': 'centimetre',
        'm': 'metre',
        'mm': 'millimetre',
        'g': 'gram',
        'gm': 'gram',
        'kg': 'kilogram',
        'mg': 'milligram',
        'ml': 'millilitre',
        'l': 'litre',
        'oz': 'ounce',
        'lb': 'pound',
        'v': 'volt',
        'w': 'watt'
    }
    return unit_mapping.get(unit, unit)


def extract_value_and_unit(text):
    # Regular expression to match a number followed by a unit
    pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)'
    match = re.search(pattern, text)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        print("Extracted:", value, unit)
        if unit in allowed_units:
            return value, unit
        normalized_unit = normalize_unit(unit)
        if normalized_unit is not None:
            return value, normalized_unit
    return None, None


output_data = []

for index, row in df.iterrows():
    idx = row['index']
    img_url = row['image_link']
    entity_name = row['entity_name']
    image = Image.open(requests.get(img_url, stream=True).raw)

    print(idx, img_url, entity_name)

    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

    generation_args = {
        "max_new_tokens": 200,
        "temperature": 0.0,
        "do_sample": False,
    }

    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]

    print(response)
    try:
        data = json.loads(response)
        if entity_name.startswith("item_"):
            print("Changed entity name")
            entity_name = entity_name[5:]
        if entity_name == "maximum_weight_recommendation":
            print("Changed entity name")
            entity_name = "weight"
        entity_value = data[entity_name]
        ans = extract_value_and_unit(entity_value)
        print("Splitted:", ans)
        if ans is not None and ans[0] is not None and ans[1] is not None:
            output_data.append({
                "index": idx,
                "prediction": f"{ans[0]} {ans[1]}",
            })
        else:
            raise Exception("Units are invalid")
    except Exception as e:
        print(idx, "")
        output_data.append({
            "index": idx,
            "prediction": "",
        })
        print(e)

output_df = pd.DataFrame(output_data)

output_df.to_csv(output_csv_path, index=False)

print(f"Data saved to {output_csv_path}")
