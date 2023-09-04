A NSFW checker for [Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). Replaces non-worksafe images with black squares. Install it from UI.

## modification

- preload model on startup.
- add nsfw switch & adjustment option.
- modify `StableDiffusionSafetyChecker`, remove black image replacement.
- add generate params with `nsfw_check` field.
