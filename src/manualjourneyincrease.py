from diffusers import StableDiffusionPipeline
import torch
import os
from PIL import Image

device = 'cuda'
seed = 6960044056360496
prompt = "(((simplistic))), concept art, stylized, splash art, symmetrical, illumination lighting, neural network design,single logo, centered, symbol, shaded, dark shadows, dynamic lighting, watercolor paint, rough paper texture, dark background, darkmode"
negative_prompt = ''
count=1
steps = 150
width = 512
height = 512

guidance_scale_list = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

files = []

for i, guidance_scale in enumerate(guidance_scale_list):
    try:
        model_id = r"models/openjourney"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        generator = torch.Generator(device)
        if not seed:
            generator.seed()
        
        outputtext = f"**Text prompt:** {prompt}\n"
        outputtext += f"**Negative text prompt:** {negative_prompt}\n"
        outputtext += f"**Count:** {count}\n"
        outputtext += f"**Seed:**  {generator.initial_seed() if not seed else seed}\n"
        outputtext += f"**Guidance scale:** {guidance_scale}\n"
        outputtext += f"**Steps:** {steps}\n"
        outputtext += f"**Size:** {width}x{height}\n"

        filename = f"{seed}_{guidance_scale}-{steps}.png"

        if seed:
            generator = generator.manual_seed(seed)

        result = pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            guidance_scale=guidance_scale,
            num_images_per_prompt=count, 
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator
        )
        if result:
            print(f"result: '{result}'")
            if result.images:
                print(f"result images: '{result.images}'")
            else:
                print("No images")
        else:
            print("No result")
        
        for i, image in enumerate(result.images):
            # If NSFW Detected
            if result.nsfw_content_detected[i] == True:
                outputtext += f"NSFW detected on image {i + 1} of {count}\n"

            name = f"{i+1}_{filename}"
            image.save(name, 'PNG')
            files.append(name)
    except RuntimeError as e:
        if 'out of CUDA out of memory' in str(e):
            outputtext += f"Out of memory: try another prompt"
    print(outputtext)
    print()

for file in files:
    print(file)