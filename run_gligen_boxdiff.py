from dataclasses import field
import random
import os
import pprint
from typing import List

import pyrallis
import torch
from PIL import Image
from config import RunConfig
from pipeline.gligen_pipeline_boxdiff import BoxDiffPipeline
from utils import ptp_utils
from utils.ptp_utils import AttentionStore

import numpy as np
from utils.drawer import draw_rectangle

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_model():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    stable_diffusion_version = "gligen/diffusers-generation-text-box"
    # If you cannot access the huggingface on your server, you can use the local prepared one.
    # stable_diffusion_version = "../../packages/diffusers/gligen_ckpts/diffusers-generation-text-box"
    stable = BoxDiffPipeline.from_pretrained(stable_diffusion_version, torch_dtype=torch.float16, safety_checker = None).to(device)

    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(prompt: List[str],
                  model: BoxDiffPipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  bbox: List[int],
                  gligen_phrases: List[str],
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:
                                                                                #這邊需要加box這個參數進來
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)

    gligen_boxes = []
    for i in range(len(bbox)):
        x1, y1, x2, y2 = bbox[i]
        gligen_boxes.append([x1/512, y1/512, x2/512, y2/512])

    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,                             #這邊需要修改
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    gligen_phrases=gligen_phrases,
                    gligen_boxes=gligen_boxes,                                  #這邊需要修改
                    gligen_scheduled_sampling_beta=0.3,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    bbox=config.bbox,
                    config=config)
    image = outputs.images[0]
    return image


@pyrallis.wrap()
def main(config: RunConfig):
    stable = load_model()
    
    category_count = {"person" : 0, "ear" : 0, "earmuffs" : 0, "face" : 0, "face-guard" : 0, "face-mask-medical" : 0, "foot" : 0, "tools" : 0, 
    "glasses" : 0, "gloves" : 0, "helmet" : 0, "hands" : 0, "head" : 0, "medical-suit" : 0, "shoes" : 0, "safety-suit" : 0, "safety-vest" : 0}
    
    controller = AttentionStore()
    controller.num_uncond_att_layers = -16
    #prompt_output_path = config.output_path / str(len(os.listdir(config.output_path)) + 1)
    prompt_output_path = config.output_path
    prompt_output_path.mkdir(exist_ok=True, parents=True)

    for info in config.info:
        tokens = [stable.tokenizer.decode(code) for code in stable.tokenizer(info['prompt'])['input_ids']]
                
        token_indices, bbox, gligen_phrases = [], [], []
        for i in range(len(tokens)):
            if tokens[i] == "ear":
                if tokens[i + 1]  == "mu":
                    if "earmuffs" in info  and category_count["earmuffs"] < len(info["earmuffs"]):
                        token_indices.append(i+1)
                        gligen_phrases.append("earmuffs")
                        bbox.append(info["earmuffs"][category_count["earmuffs"]])
                        category_count["earmuffs"] += 1
                elif "ear" in info  and category_count["ear"] < len(info["ear"]):
                    token_indices.append(i)
                    gligen_phrases.append("ear")
                    bbox.append(info["ear"][category_count["ear"]])
                    category_count["ear"] += 1
                    
            elif tokens[i] == "face":
                if tokens[i + 1] != "-":
                    if "face" in info  and category_count["face"] < len(info["face"]):
                        token_indices.append(i)
                        gligen_phrases.append("face")
                        bbox.append(info["face"][category_count["face"]])
                        category_count["face"] += 1
                elif tokens[i + 2] == "guard":
                    if  "face-guard" in info  and category_count["face-guard"] < len(info["face-guard"]):
                        token_indices.append(i+2)
                        gligen_phrases.append("face-guard")
                        bbox.append(info["face-guard"][category_count["face-guard"]])
                        category_count["face-guard"] += 1
                elif "face-mask-medical" in info  and category_count["face-mask-medical"] < len(info["face-mask-medical"]):
                    token_indices.append(i+2)
                    gligen_phrases.append("face-mask-medical")
                    bbox.append(info["face-mask-medical"][category_count["face-mask-medical"]])
                    category_count["face-mask-medical"] += 1
                    
            elif tokens[i] == "medical":
                if "medical-suit" in info  and category_count["medical-suit"] < len(info["medical-suit"]):
                    token_indices.append(i+2)
                    gligen_phrases.append("medical-suit")
                    bbox.append(info["medical-suit"][category_count["medical-suit"]])
                    category_count["medical-suit"] += 1
                
            elif tokens[i]  == "safety":
                if tokens[i + 1]  != "-":
                    continue
                if tokens[i + 2] == "suit":
                    if "safety-suit" in info  and category_count["safety-suit"] < len(info["safety-suit"]):
                        token_indices.append(i+2)
                        gligen_phrases.append("safety-suit")
                        bbox.append(info["safety-suit"][category_count["safety-suit"]])
                        category_count["safety-suit"] += 1
                elif "safety-vest" in info  and category_count["safety-vest"] < len(info["safety-vest"]):
                    token_indices.append(i+2)
                    gligen_phrases.append("safety-vest")
                    bbox.append(info["safety-vest"][category_count["safety-vest"]])
                    category_count["safety-vest"] += 1
                    
            elif tokens[i] in info and category_count[tokens[i]] < len(info[tokens[i]]):
                token_indices.append(i)
                gligen_phrases.append(tokens[i])
                bbox.append(info[tokens[i]][category_count[tokens[i]]])
                category_count[tokens[i]] += 1
                
        config.bbox = bbox
        config.prompt = info['prompt']
                
        print(f"Current iamge is : {info['name']}")

        config.seeds = [random.randint(0, 1000000)]  # Generates a list of 5 random seeds
        for seed in config.seeds:
            print(f"Current seed is : {seed}")
            g = torch.Generator('cuda').manual_seed(seed)
            
            image = run_on_prompt(prompt=info['prompt'],
                                  gligen_phrases=gligen_phrases,
                                  model=stable,
                                  controller=controller,
                                  token_indices=token_indices,
                                  bbox=bbox,
                                  seed=g,
                                  config=config)
            
            image.save(prompt_output_path / info['name'])
                
        for category in category_count:
            category_count[category] = 0
            
if __name__ == '__main__':
    main()
