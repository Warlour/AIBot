from diffusers import StableDiffusionPipeline
import torch
device = 'cuda'
prompt = ''
negative_prompt = ''
count = 3
seed = 1
guidance_scale = 10

filename = "prompt.png"
files = []

try:
    model_id = r"models/openjourney"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    generator = torch.Generator(device)
    if not seed:
        generator.seed()
    outputtext = f"**Text prompt:** {prompt}\n**Negative text prompt:** {negative_prompt}\n**Count:** {count}\n**Seed:**  {generator.initial_seed()}\n"

    result = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        guidance_scale=guidance_scale, 
        num_images_per_prompt=count, 
        generator=generator.manual_seed(int(seed))
    ) if seed else pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        guidance_scale=guidance_scale, 
        num_images_per_prompt=count
    )

    for i, image in enumerate(result.images):
        # If NSFW Detected
        if result.nsfw_content_detected[i] == True:
            outputtext += f"NSFW detected on image {i + 1} of {count}\n"

        name = f"{i+1}_{filename}"
        image.save(name, 'PNG')
except RuntimeError as e:
    if 'out of CUDA out of memory' in str(e):
        outputtext += f"Out of memory: try another prompt"

print(outputtext)