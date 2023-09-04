import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoFeatureExtractor

from modules import scripts

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
    has_nsfw_concept = safety_checker(clip_input=safety_checker_input.pixel_values)

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
        images = kwargs['images']
        has_nsfw_concept = censor_batch(images)[:]
        for i, x in enumerate(has_nsfw_concept):
            if x:
                image = Image.fromarray((images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                image = image.filter(ImageFilter.GaussianBlur(radius=50))
                images[i] = torch.from_numpy(np.array(image)/255).permute(2, 0, 1)

        p.extra_generation_params.update({"nsfw_check": has_nsfw_concept})
