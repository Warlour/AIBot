import discord, os, requests, asyncio
from discord import app_commands

discord_token = os.environ["DISCORD_TOKEN_BOT1"]

guildObject = discord.Object(id = 1084203391519051786)

class aclient(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.synced = False
    
    async def on_ready(self):
        await self.wait_until_ready()
        if not self.synced:
            await tree.sync(guild = guildObject)
            self.synced = True

        print(f"We have logged in as {self.user}")

client = aclient()
tree = app_commands.CommandTree(client)

# Default values
pygmalionState = False
pygmalionCharacter = None

@tree.command(name = "test", description = "Used for testing purposes", guild = guildObject)
async def self(interaction: discord.Interaction):
    #await interaction.response.defer()
    modal = discord.ui.Modal(title="Create ...", timeout=None)

    urlInput = discord.ui.TextInput(label="URL", placeholder="Insert YouTube link here!", required=False)
    modal.add_item(urlInput)
    
    # transcribeOpt = discord.SelectOption(label="Transcribe", description="Transcribes the audio to text in English", default=True)
    # detectOpt = discord.SelectOption(label="Detect Language", description="Detects the language spoken in the audio", default=False)
    # options = [transcribeOpt, detectOpt]
    # optSelect = discord.ui.Select(min_values=1, max_values=1, options=options)
    # modal.add_item(optSelect)

    await interaction.response.send_modal(modal)

@tree.command(name = "clear", description = "Clear the current channel", guild = guildObject)
async def self(interaction: discord.Interaction):
    await interaction.response.defer()
    if interaction.user.id == 152917625700089856:
        await interaction.channel.purge()
    else:
        interaction.response.send_message(content="You do not have permissions to do this!")

'''Custom'''
from logic import *

'''AI general'''
device = 'cuda'
import torch

'''Stable-diffusion | Image generation'''
stbdfsPath = r"models/stable-diffusion-v1-5"
from diffusers import StableDiffusionPipeline

@tree.command(name = "t2i", description="Generate text to image using Stable Diffusion v1.5", guild = guildObject)
async def self(interaction: discord.Interaction, prompt:str, negative_prompt:str = None, count:int = 1, seed:int = None):
    # if not interaction.channel.id == 1090783286793621614:
    #     await interaction.response.send_message("You're in the wrong channel!")
    #     await interaction.delete_original_response()

    await interaction.response.defer()

    if count < 1 or count > 5:
        await interaction.response.followup.send(content="I cannot send less than 1 or more than 5 pictures!")
        return
    
    if not prompt:
        await interaction.response.followup.send(content="No prompt given")

    filename = f"prompt.png"

    files = []

    try:
        model_id = stbdfsPath
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        generator = torch.Generator(device)
        if not seed:
            generator.seed()
        outputtext = f"**Text prompt:** {prompt}\n**Count:** {count}\n**Seed:**  {generator.initial_seed()}\n"

        result = pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            num_images_per_prompt=count, 
            generator=generator.manual_seed(int(seed))
        ) if seed else pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            num_images_per_prompt=count, 
            generator=generator
        )
        
        for i, image in enumerate(result.images):
            # If NSFW Detected
            if result.nsfw_content_detected[i] == True:
                outputtext += f"NSFW Detected on image {i + 1} of {count}\n"

            name = f"{i+1}_{filename}"
            image.save(name, 'PNG')
            files.append(discord.File(fp=name, description=f"Image {i + 1} of {count}"))
    except RuntimeError as e:
        if 'out of CUDA out of memory' in str(e):
            outputtext += f"Out of memory: try another prompt"

    await interaction.followup.send(content=outputtext, files=files)

    for file in files:
        os.remove(file.filename)

'''OpenJourney'''
@tree.command(name = "openjhelp", description="Get help with OpenJourney", guild = guildObject)
async def self(interaction: discord.Interaction):
    message = "**Prompt:** the text that the model with generate an image with\n"
    message += "**Negative prompt:** the text that the model with avoid generating an image with\n"
    message += "**Count:** amount of images to generate\n"
    message += "**Seed:** the seed to use when generating the image\n"
    message += "**Guidance scale:** how similar the image is to the prompt\n"
    message += "**Steps:** More steps = more quality and time to generate\n"
    message += "**Width and height:** image dimensions\n"
    await interaction.response.send_message(message)

@tree.command(name = "openjourney", description="Generate text to image using OpenJourney", guild = guildObject)
async def self(interaction: discord.Interaction, prompt:str, negative_prompt:str = None, guidance_scale:float = 7.5, count:int = 1, seed:int = None, steps:int = 50, width:int = 512, height:int = 512):
    await interaction.response.defer()

    if count < 1 or count > 5:
        await interaction.response.followup.send(content="I cannot generate less than 1 or more than 5 pictures!")
        return
    
    if not prompt:
        await interaction.response.followup.send(content="No prompt given")

    filename = "output.png"

    files = []

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

        result = pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            guidance_scale=guidance_scale,
            num_images_per_prompt=count, 
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator.manual_seed(int(seed))
        ) if seed else pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            guidance_scale=guidance_scale,
            num_images_per_prompt=count, 
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator
        )
        print(result.images)
        
        for i, image in enumerate(result.images):
            # If NSFW Detected
            if result.nsfw_content_detected[i] == True:
                outputtext += f"NSFW detected on image {i + 1} of {count}\n"

            name = f"{i+1}_{filename}"
            image.save(name, 'PNG')
            files.append(discord.File(fp=name, description=f"Image {i + 1} of {count}"))
    except RuntimeError as e:
        if 'out of CUDA out of memory' in str(e):
            outputtext += f"Out of memory: try another prompt"

    await interaction.followup.send(content=outputtext, files=files)

    for file in files:
        os.remove(file.filename)

from diffusers import StableDiffusionImg2ImgPipeline

@tree.command(name = "i2i", description="Generate image to image using Stable Diffusion v1.5", guild = guildObject)
async def self(interaction: discord.Interaction, prompt:str, file: discord.Attachment, negative_prompt:str = None, count:int = 1, seed:int = None):
    # if not interaction.channel.id == 1084511996139020460:
    #     await interaction.response.send_message("You're in the wrong channel!")
    #     await interaction.delete_original_response()

    await interaction.response.defer()

    if count < 1 or count > 5:
        await interaction.followup.send(content="I cannot send less than 1 or more than 5 pictures!")
        return
    
    if not prompt:
        await interaction.followup.send(content="No prompt given")
    
    if not file or not file.filename.endswith(('.png', '.jpg', '.webp', 'jpeg')):
        await interaction.followup.send(content="Invalid file extension")
        return
    
    r = requests.get(file.url)
    with open(file.filename, 'wb') as f:
        f.write(r.content)
    
    filename = file.filename

    files = []
    files.append(discord.File(fp=filename, description="Prompt file"))

    try:
        model_id = stbdfsPath
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        generator = torch.Generator(device)
        if not seed:
            generator.seed()
        outputtext = f"**Text prompt:** {prompt}\n**Count:** {count}\n**Seed:**  {generator.initial_seed()}\n"

        with Image.open(filename) as im:
            result = pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt, 
                image=im, 
                num_images_per_prompt=count, 
                generator=generator.manual_seed(int(seed))
            ) if seed else pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt, 
                image=im, 
                num_images_per_prompt=count, 
                generator=generator
            )
        
        for i, image in enumerate(result.images):
            # If NSFW Detected
            if result.nsfw_content_detected[i] == True:
                outputtext += f"NSFW Detected on image {i + 1} of {count}\n"

            name = f"{i+1}_{filename}"
            image.save(name, 'PNG')
            files.append(discord.File(fp=name, description=f"Image {i + 1} of {count}"))
    except RuntimeError as e:
        if 'out of CUDA out of memory' in str(e):
            outputtext += f"Out of memory: try another prompt"

    await interaction.followup.send(content=outputtext, files=files)

    for file in files:
        os.remove(file.filename)

'''Whisper | Transcription of audio and video'''
import whisper

import validators

# Attach audio file and output text
@tree.command(name = "whisper", description="Generate transcriptions and detect language using OpenAI's Whisper model", guild = guildObject)
async def self(interaction: discord.Interaction, file:discord.Attachment = None, url:str = None, transcribe:bool = True, prompt:str = "", detect:bool = False):
    # if not interaction.channel.id == 1084408319457894400:
    #     await interaction.response.send_message("You're in the wrong channel!")
    #     await interaction.delete_original_response()

    await interaction.response.defer()

    if not transcribe and not detect:
        await interaction.followup.send(content="No operation given; use transcribe and/or detect!")
        return
    
    if not file and not url:
        await interaction.followup.send(content="No file or url attached")
    
    if file and url:
        await interaction.followup.send(content="You can only add a file __or__ an url!")

    if file:
        if not file.filename.endswith(('.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm')):
            await interaction.followup.send(content="Invalid file extension")
            return
        
        print(f"Downloading {file.filename}")
        r = requests.get(file.url)
        with open(file.filename, 'wb') as f:
            f.write(r.content)
        
        filename = file.filename
    elif url:
        if (validators.url(url)):
            filename = ytdownload(url)
            print(filename)
        else:
            await interaction.followup.send(content="Invalid url!")
            return

    model_name = "medium"
    model = whisper.load_model(model_name, device=device)

    output = ""
    if detect:
        audio = whisper.load_audio(filename)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)
        output += f"Detected language: {max(probs, key=probs.get)}"
    if transcribe:
        result = model.transcribe(filename, initial_prompt=prompt) if prompt else model.transcribe(filename)
        if detect:
            output += " | "
        output += f"Transcribed {filename}"

    if result['text']:
        inputPrompt = discord.File(fp=filename)

        with open(f"transcription_{filename}.txt", "w") as f:
            f.write(result['text'].strip())
        outputfile = discord.File(fp=f"transcription_{filename.rsplit(sep='.', maxsplit=1)}.txt")

        files = [inputPrompt, outputfile]
        await interaction.followup.send(content=output, files=files)
    else:
        await interaction.followup.send(content="Could not create a transcription")

    for file in files:
        os.remove(file.filename)

'''Clip | Guessing'''
import clip
import numpy as np

# Attach image and output text
@tree.command(name = "clip", description="Attach an image and possible guesses to make AI guess what is in image", guild = guildObject)
async def self(interaction: discord.Interaction, file:discord.Attachment, prompt:str):
    # if not interaction.channel.id == 1084408335899566201:
    #     await interaction.response.send_message("You're in the wrong channel!")
    #     await interaction.delete_original_response()

    await interaction.response.defer()

    if not file:
        await interaction.followup.send(content="No file attached!")
        return
    
    if not file.filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        await interaction.followup.send(content="Invalid file extension")
        return
    
    if not prompt:
        await interaction.followup.send(content="No prompt given!")
        return

    print(f"Downloading {file.filename}")
    r = requests.get(file.url)
    with open(file.filename, 'wb') as f:
        f.write(r.content)
    
    filename = file.filename

    model_name = "ViT-B/32"

    # Load model
    model, preprocess = clip.load(model_name, device=device)

    image = preprocess(Image.open(filename)).unsqueeze(0).to(device)
    possibilities = prompt.split(", ")
    textprob = clip.tokenize(possibilities).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(textprob)
        logits_per_image, logits_per_text = model(image, textprob)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Create list of percentages
    probas = []
    for item in probs[0]:
        probas.append(float(np.format_float_positional(item, precision=4)))
    
    # Create and sort list with possibilites and percentages combined
    list_sorted = sorted(zip(possibilities, probas), key=lambda x: x[1], reverse=True)
    print(list_sorted)

    # Format list
    text_list = []
    for item in list_sorted:
        text_list.append(f"{item[0]}: {item[1] * 100:.2f}%")
    output = "\n".join(text_list)
    print(text_list)
    
    # Send output to discord
    if output:
        imagePrompt = discord.File(fp=filename)

        with open(f"guess_{filename}.txt", "w") as f:
            f.write(output)
        outputfile = discord.File(fp=f"guess_{filename}.txt")

        files = [imagePrompt, outputfile]
        await interaction.followup.send(content="", files=files)
        
        # Remove files
        for file in files:
            os.remove(file.filename)

@tree.command(name = "pygmalion", description="Conversational bot set state with character", guild = guildObject)
async def self(interaction: discord.Interaction, state:bool, character:str = None):
    await interaction.response.defer()
    if interaction.channel.id != 1091464570943574076:
        channel = client.get_channel(1091464570943574076)
        interaction.followup.send(content=f"Due to it being a conversational bot, this bot is only available in the {channel} channel.")
        return
    global pygmalionState, pygmalionCharacter
    pygmalionState = state
    pygmalionCharacter = character

from transformers import AutoTokenizer, GPTJForCausalLM

@client.event
async def on_message(ctx: discord.Message):
    if ctx.channel.id != 1091464570943574076:
        return
    
    if not pygmalionState:
        await ctx.channel.send(content="Pygmalion is not active!")
        return
    
    if not pygmalionCharacter:
        await ctx.channel.send(content="No character to play!")
        return

    try:
        model_id = r"models/pygmalion-6b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = GPTJForCausalLM.from_pretrained(model_id)

        messages = await ctx.channel.history(limit=200)
        print(messages)

        history = []
        for msg in messages:
            pass

    except:
        pass
    # Testing
    if True:
        messages = ctx.channel.history(limit=200)
        for message in messages:
            print(message)

client.run(discord_token)