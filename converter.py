import json
import os

path_in = "./inputs"
path_out = "./outputs"

with open(f"{path_in}/prompts.json", "r") as file:
    prompts = json.load(file)

list = []

categories = ["person", "ear", "earmuffs", "face", "face-guard", "face-mask-medical", "foot", "tools", 
              "glasses", "gloves", "helmet", "hands", "head", "medical-suit", "shoes", "safety-suit", "safety-vest"]

for filename in prompts:
    filename_split = filename.rsplit('.', 1)[0]
    label_path = f'{path_in}/labels/{filename_split}.txt'
        
    width, height = 512, 512
    dict = {"name" : filename, "prompt" : prompts[filename] + " There are "}
        
    with open(label_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            data = [float(num) for num in line.split()]
            w, h = int(width * data[3]), int(height * data[4])
            x1, y1 = width * (data[1] - data[3] / 2), height * (data[2] - data[4] / 2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
            dict.setdefault(categories[int(data[0])], []).append([x1, y1, x2, y2])
            dict['prompt'] += f"{categories[int(data[0])]}" if i == 0 else f", {categories[int(data[0])]}"

    dict['prompt'] += ". 8k resolution, highly detailed, ultra-realistic, anatomically correct, well-defined hands and fingers."
    list.append(dict)

with open(f'{path_out}/prompts_with_info.json', 'w') as f:
    json.dump(list, f)
