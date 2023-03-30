import discord, os, requests, asyncio
from discord import app_commands

discord_token = os.environ["DISCORD_TOKEN_BOT1"]

class aclient(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.synced = False
    
    async def on_ready(self):
        await self.wait_until_ready()
        if not self.synced:
            await tree.sync(guild = discord.Object(id = 1084203391519051786))
            self.synced = True
        print(f"We have logged in as {self.user}.")

client = aclient()
tree = app_commands.CommandTree(client)

@tree.command(name = "clear", description = "Clear the current channel", guild = discord.Object(id = 1084203391519051786))
async def self(interaction: discord.Interaction):
    await interaction.response.defer()
    await interaction.channel.purge()

'''Custom'''
from logic import *

'''Processing'''
from io import BytesIO

'''AI general'''
device = 'cuda'
import torch

'''Stable-diffusion | Image generation'''
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline

@tree.command(name = "t2i", description="Generate text to image using Stable Diffusion", guild = discord.Object(id = 1084203391519051786))
async def self(interaction: discord.Interaction, prompt:str, count:int = 1, seed:int = None):
    if not interaction.channel.id == 1084440260517298196:
        await interaction.response.send_message("You're in the wrong channel!")
        await interaction.delete_original_response()

    # await interaction.response.defer()
    if count < 1 or count > 5:
        await interaction.response.send_message(content="I cannot send less than 1 or more than 5 pictures!")
        return

    if not prompt:
        await interaction.response.send_message(content="No prompt given")

    await interaction.response.send_message(f"**Prompt:** {prompt}\n**Count:** {count}\n**Seed:** {seed if seed else 'random'}")
    for _ in range(int(count)):
        print(f"{prompt} | Image nr: {str(_)}")
        try:
            pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-4", torch_dtype=torch.float16).to(device)
            result = pipe(prompt, generator=torch.Generator(device).manual_seed(int(seed))) if seed else pipe(prompt)
            if result['nsfw_content_detected'] == [True]:
                await interaction.followup.send(content=f"NSFW Detected on image {_ + 1} of {int(count)}")
                raise ValueError('NSFW Detected')
            
            with BytesIO() as image_binary:
                result.images[0].save(image_binary, 'PNG')
                image_binary.seek(0)
                await interaction.followup.send(content=f"Image {_ + 1} of {int(count)}", file=discord.File(fp=image_binary, filename='image.png'))
        except ValueError as e:
            if 'NSFW Detected' in str(e):
                pass
            else:
                print(e)

@tree.command(name = "t2i2", description="Generate text to image using Stable Diffusion v1.5", guild = discord.Object(id = 1084203391519051786))
async def self(interaction: discord.Interaction, prompt:str, count:int = 1, seed:int = None):
    if not interaction.channel.id == 1090783286793621614:
        await interaction.response.send_message("You're in the wrong channel!")
        await interaction.delete_original_response()

    if count < 1 or count > 5:
        await interaction.response.send_message(content="I cannot send less than 1 or more than 5 pictures!")
        return
    
    if not prompt:
        await interaction.response.send_message(content="No prompt given")
    
    await interaction.response.send_message(f"**Prompt:** {prompt}\n**Count:** {count}\n**Seed:** {seed if seed else 'random'}")
    for _ in range(int(count)):
        print(f"{prompt} | Image nr: {str(_)}")
        try:
            pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
            result = pipe(prompt, generator=torch.Generator(device).manual_seed(int(seed))) if seed else pipe(prompt)
            if result['nsfw_content_detected'] == [True]:
                await interaction.followup.send(content=f"NSFW Detected on image {_ + 1} of {int(count)}")
                raise ValueError('NSFW Detected')
            
            with BytesIO() as image_binary:
                result.images[0].save(image_binary, 'PNG')
                image_binary.seek(0)
                await interaction.followup.send(content=f"Image {_ + 1} of {int(count)}", file=discord.File(fp=image_binary, filename='image.png'))
        except ValueError as e:
            if 'NSFW Detected' in str(e):
                pass
            else:
                print(e)

@tree.command(name = "i2i", description="Generate image to image using Stable Diffusion v1.5", guild = discord.Object(id = 1084203391519051786))
async def self(interaction: discord.Interaction, prompt:str, file: discord.Attachment, count:int = 1, seed:int = None):
    if not interaction.channel.id == 1084511996139020460:
        await interaction.response.send_message("You're in the wrong channel!")
        await interaction.delete_original_response()

    if count < 1 or count > 5:
        await interaction.response.send_message(content="I cannot send less than 1 or more than 5 pictures!")
        return
    
    if not prompt:
        await interaction.response.send_message(content="No prompt given")
    
    if not file or not file.filename.endswith(('.png', '.jpg', '.webp', 'jpeg')):
        await interaction.response.send_message(content="Invalid file extension")
        return
    
    r = requests.get(file.url)
    with open(file.filename, 'wb') as f:
        f.write(r.content)

    # with open(file.filename, 'rb') as f:
    #     file = discord.File(f)
    #     await interaction.response.send_message(content=f"**Text prompt:** {prompt}\n**Count:** {count}\n**Seed:** {seed if seed else 'random'}", file=file)
    await interaction.response.defer()

    image_prompt = Image.open(file.filename)
    for _ in range(int(count)):
        print(f"{prompt} | Image nr: {str(_)}")
        try:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained("stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
            result = pipe(prompt=prompt, image=image_prompt, generator=torch.Generator(device).manual_seed(int(seed))) if seed else pipe(prompt=prompt, image=image_prompt)

            if result['nsfw_content_detected'] == [True]:
                await interaction.followup.send(content=f"NSFW Detected on image {_ + 1} of {int(count)}")
                raise ValueError('NSFW Detected')
            
            with BytesIO() as image_binary:
                result.images[0].save(image_binary, 'PNG')
                image_binary.seek(0)
                await interaction.followup.send(content=f"Image {_ + 1} of {int(count)}", file=discord.File(fp=image_binary, filename='image.png'))
        except ValueError as e:
            if 'NSFW Detected' in str(e):
                pass
            else:
                print(e)
        except RuntimeError as e:
            if 'out of CUDA out of memory' in str(e):
                interaction.followup.send(content=f"Out of memory: Image {_ + 1} of {int(count)}, try another image")
    # try:
    image_prompt.close()
    os.remove(file.filename)
    # except:
    #     pass

'''Whisper | Transcription of audio and video'''
import whisper

import validators

@tree.command(name = "whisper", description="Generate transcriptions and detect language using OpenAI's Whisper model", guild = discord.Object(id = 1084203391519051786))
async def self(interaction: discord.Interaction, file: discord.Attachment = None, url:str = None, transcribe:bool = True, prompt:str = "", detect:bool = False):
    if not interaction.channel.id == 1084408319457894400:
        await interaction.response.send_message("You're in the wrong channel!")
        await interaction.delete_original_response()

    await interaction.response.defer()

    if not transcribe and not detect:
        await interaction.response.send_message(content="No operation given; use transcribe and/or detect!")
        return
    
    if not file and not url:
        await interaction.response.send_message(content="No file or url attached")
    
    if file and url:
        await interaction.response.send_message(content="You can only add a file __or__ an url!")

    if file:
        if not file.filename.endswith(('.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm')):
            await interaction.response.send_message(content="Invalid file extension")
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
            await interaction.response.send_message(content="Invalid url!")
            return

    # TODO: Send file without permissionerror exception
    # with open(filename, 'rb') as f:
    #     file = discord.File(f)
    #     await interaction.response.send_message(content=f"**Text prompt:** {prompt}\n**Count:** {count}\n**Seed:** {seed if seed else 'random'}", file=file)

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
        output += f"Transcribed {filename}\n`{result['text'].strip()}`"
    
    await interaction.followup.send(content=output)
    os.remove(filename)

client.run(discord_token)