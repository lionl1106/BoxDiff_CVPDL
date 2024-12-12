import json
import os

path_in = "/content/inputs/info.json"
path_prompts = "/content/inputs/prompts.json"
path_out = "/content/inputs"

with open(f"{path_prompts}", "r") as file:
    prompts = json.load(file)

with open(f"{path_in}", "r") as file:
    images = json.load(file)

list = []

for image in images:
    dict = {"name" : image['image'], "prompt" : prompts[image['image']] + " There are "}
        
    for i, obj in enumerate(image['labels']):
        obj = obj.lower()
        dict.setdefault(obj, []).append(image['bboxes'][i])
        dict['prompt'] += f"{obj}" if i == 0 else f", {obj}"

    dict['prompt'] += ". 8k resolution, highly detailed, ultra-realistic, anatomically correct, well-defined hands and fingers."
    list.append(dict)

os.makedirs(path_out, exist_ok=True)
with open(f'{path_out}/prompts_with_info.json', 'w') as f:
    json.dump(list, f)
