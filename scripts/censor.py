import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoFeatureExtractor

from modules import scripts, shared, script_callbacks

from safety_checker.safety_checker import StableDiffusionSafetyChecker

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    has_nsfw_concept = safety_checker(clip_input=safety_checker_input.pixel_values, adjustment_init=shared.opts.filter_nsfw_adjustment)

    return has_nsfw_concept


def censor_batch(x):
    x_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
    has_nsfw_concept = check_safety(x_samples_ddim_numpy)
    return has_nsfw_concept


class NsfwCheckScript(scripts.Script):
    def title(self):
        return "NSFW check"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        if shared.opts.filter_nsfw is False:
            return
        
        images = kwargs['images']
        has_nsfw_concept = censor_batch(images)[:]
        for i, x in enumerate(has_nsfw_concept):
            if x:
                image = Image.fromarray((images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                image = image.filter(ImageFilter.GaussianBlur(radius=50))
                images[i] = torch.from_numpy(np.array(image)/255).permute(2, 0, 1)

        if "nsfw_check" not in p.extra_generation_params:
            p.extra_generation_params["nsfw_check"] = []
        p.extra_generation_params["nsfw_check"].extend(has_nsfw_concept)


def on_ui_settings():
    import gradio as gr
    shared.opts.add_option("filter_nsfw", shared.OptionInfo(True, "Filter NSFW", gr.Checkbox, {"interactive": True}, section=("nsfw", "NSFW")))
    shared.opts.add_option("filter_nsfw_adjustment", shared.OptionInfo(-0.04, "Filter NSFW adjustment", gr.Slider, {"interactive": True, "minimum": -0.5, "maximum": 0.5, "step": 0.01}, section=("nsfw", "NSFW Adjustment")))

script_callbacks.on_ui_settings(on_ui_settings)