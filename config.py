from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import random
import json


@dataclass
class RunConfig:
    # Guiding text prompt
    prompt: str = ""
    
    # Whether to use Stable Diffusion v2.1
    sd_2_1: bool = False
    
    # Which token indices to alter with attend-and-excite
    token_indices: List[int] = None
    
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [random.randint(0, 10000) for _ in range(2)])
    
    # Path to save all outputs to
    output_path: Path = Path('/content/outputs')
    
    # Number of denoising steps
    n_inference_steps: int = 70
    
    # Text guidance scale
    guidance_scale: float = 7.5
    
    # Number of denoising steps to apply attend-and-excite
    max_iter_to_alter: int = 25
    
    # Resolution of UNet to compute attention maps over
    attention_res: int = 16
    
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(default_factory=lambda: {0: 0.05, 10: 0.5, 20: 0.8})
    
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 20
    
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))
    
    # Whether to apply the Gaussian smoothing before computing the maximum attention value for each subject token
    smooth_attentions: bool = True
    
    # Standard deviation for the Gaussian smoothing
    sigma: float = 0.5
    
    # Kernel size for the Gaussian s
    # moothing
    kernel_size: int = 3
    
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False
    
    # BoxDiff
    bbox: List[list] = field(default_factory=lambda: [[], []])
    color: List[str] = field(default_factory=lambda: ['blue', 'red', 'purple', 'orange', 'green', 'yellow', 'black'])
    P: float = 0.2
    
    # number of pixels around the corner to be selected
    L: int = 1
    refine: bool = True
    gligen_phrases: List[str] = field(default_factory=lambda: ['', ''])
    n_splits: int = 4
    which_one: int = 1
    eval_output_path: Path = Path('./outputs/eval')
    
    with open(f"/content/inputs/prompts_with_info.json", "r") as file:
      info = json.load(file)

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)
