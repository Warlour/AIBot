import os
import discord
import requests

# Processing
import numpy as np
import textwrap
import re

from logic import *

# Models and related
'''Whisper | Transcription of audio and video'''
import whisper
'''CLIP | Image guesser'''
import torch
import clip
'''Stable-diffusion | Image generation'''
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
'''Pygmalion-6b | Character player'''
from transformers import AutoTokenizer, AutoModelForCausalLM

#openai.api_key = os.environ["OPENAI_API_KEY"]
discord_token = os.environ["DISCORD_TOKEN_BOT1"]

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

device = "cuda" if torch.cuda.is_available() else "cpu"

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

# If you want the models to work on your own Discord server, you have to change the channel ids
@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content == 'clear' and message.author.id == 152917625700089856:
        print(f"Purging channel: {message.channel.name}")
        await message.channel.purge()

    '''Models only requiring text'''
    # Stable diffusion -> Image
    if message.channel.id == 1084440260517298196:
        # Any number followed by a colon
        pattern = r'^\d+:'

        # If pattern matches message
        if re.match(pattern, message.content):
            # Split string to number and message
            splitstring = re.split(r':', message.content, maxsplit=1)

            # Number
            x = splitstring[0]
            if int(x) < 1 or int(x) > 5:
                return

            # Prompt
            prompt = splitstring[-1]
        else:
            return
        
        await message.add_reaction('⏳')

        # Generate x images
        for _ in range(int(x)):
            print(f"{prompt} | Image nr: {str(_)}")
            try:
                pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-4", torch_dtype=torch.float16).to(device)

                filename = f"{_}{prompt}.png"
                pipe(prompt.strip()).images[0].save(filename)

                with open(filename, 'rb') as f:
                    file = discord.File(f)
            
                if is_image(filename):
                    await message.channel.send(f"{prompt} | Image {_ + 1} of {int(x)}", file=file)

                    if _ == int(x)-1:
                        await message.remove_reaction('⏳', client.user)
                        await message.add_reaction('✅')
            except Exception as e:
                print(e)
                await message.remove_reaction('⏳', client.user)
                await message.add_reaction('❌')

            f.close()
            os.remove(filename)

    # Pygmalion
    if message.channel.id == 1090751089059573871:
        pass

    '''Models requiring files as input'''
    for attachment in message.attachments:
        # Download and write file as binary
        r = requests.get(attachment.url)
        with open(attachment.filename, 'wb') as f:
            f.write(r.content)

        await message.add_reaction('⏳')

        try:
            # Whisper -> Text
            if message.channel.id == 1084408319457894400 and attachment.filename.endswith(('mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm')):
                if not any(string2 in message.content for string2 in ['transcribe', 'detect']):
                    try:
                        f.close()
                        os.remove(attachment.filename)
                    except:
                        pass
                    
                    await message.remove_reaction('⏳', client.user)
                    await message.add_reaction('❌')
                    await message.channel.send("I'm not sure what you want me to do?")
                    return
                
                model_name = "medium"
                model = whisper.load_model(model_name, device=device)
                # print(
                #     f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
                #     f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
                # )

                text = f"Using the {model_name} Whisper model with {sum(np.prod(p.shape) for p in model.parameters()):,} parameters:\n"
                if 'transcribe' in message.content:
                    prompt = ""
                    if 'transcribe: ' in message.content:
                        prompt = text_between(message.content, "transcribe: ", " detect")
                        print(prompt)
                    result = model.transcribe(attachment.filename, initial_prompt=prompt)
                    text += f'Transcribed {attachment.filename}: `{result["text"].strip()}`\n\n'

                if 'detect' in message.content:
                    audio = whisper.load_audio(attachment.filename)
                    audio = whisper.pad_or_trim(audio)

                    mel = whisper.log_mel_spectrogram(audio).to(model.device)

                    _, probs = model.detect_language(mel)
                    text += f"Detected language: {max(probs, key=probs.get)}\n\n"

            # CLIP -> Text
            if message.channel.id == 1084408335899566201 and attachment.filename.endswith(('.png', '.jpg', '.jpeg', '.webp')) and message.content:
                model_name = "ViT-B/32"
                # Load model
                model, preprocess = clip.load(model_name, device=device)
                text = f"Using the {model_name} CLIP model with {sum(np.prod(p.shape) for p in model.parameters()):,} parameters:\n"

                image = preprocess(Image.open(attachment.filename)).unsqueeze(0).to(device)
                possibilities = message.content.split(", ")
                textprob = clip.tokenize(possibilities).to(device)

                with torch.no_grad():
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(textprob)
                    logits_per_image, logits_per_text = model(image, textprob)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                probas = []
                for item in probs[0]:
                    probas.append(float(np.format_float_positional(item, precision=4)))
                
                list_sorted = sorted(zip(possibilities, probas), key=lambda x: x[1], reverse=True)

                text_list = []
                for item in list_sorted:
                    text_list.append(f"{item[0]}: {item[1] * 100:.2f}%")
                text = "\n".join(text_list)

            # Stable-diffusion -> Image
            if message.channel.id == 1084511996139020460 and attachment.filename.endswith(('.png', '.jpg', '.webp')) and message.content:
                await message.add_reaction('⏳')

                pipe = StableDiffusionImg2ImgPipeline.from_pretrained("stable-diffusion-v1-4", torch_dtype=torch.float16).to(device)
                prompt = message.content
                image_prompt = Image.open(attachment.filename)
                filename = f"edited_{attachment.filename}"
                pipe(prompt=prompt, image=image_prompt).images[0].save(filename)

                with open(filename, 'rb') as f:
                    file = discord.File(f)

                if is_image(filename):
                    await message.remove_reaction('⏳', client.user)
                    await message.add_reaction('✅')
                    await message.channel.send(prompt, file=file)
                else:
                    await message.remove_reaction('⏳', client.user)
                    await message.add_reaction('❌')
        except Exception as e:
            print(e)
            # Error
            await message.remove_reaction('⏳', client.user)
            await message.add_reaction('❌')
            return
        
        # Attempt to delete attachment
        try:
            await message.remove_reaction('⏳', client.user)
            f.close()
            os.remove(attachment.filename)
        except:
            pass

        # If there is a text prompt
        if text:
            try:
                # If message is too long
                if len(text) >= 2000:
                    # Set max size
                    send_queue = textwrap.wrap(text, width=2000)

                    # Send message separately
                    for i, string in enumerate(send_queue):
                        string = string.replace("% ", "%\n")
                        if i == 0:
                            await message.channel.send(string + "`")
                        else:
                            await message.channel.send("`" + string + "`")
                # If message is not too long
                else:
                    await message.channel.send(text)
                
                # Done
                await message.add_reaction('✅')
            except UnboundLocalError:
                pass
            except discord.errors.HTTPException:
                await message.channel.send("Reached character limit of 2000.")
                await message.add_reaction('⚠️')

client.run(discord_token)
