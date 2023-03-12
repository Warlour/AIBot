import os
import discord
import requests

# Processing
from PIL import Image
import numpy as np
import textwrap

# Models and related
'''Whisper'''
import whisper
'''CLIP'''
import torch
import clip
'''Stable-diffusion'''
from diffusers import StableDiffusionPipeline

#openai.api_key = os.environ["OPENAI_API_KEY"]
discord_token = os.environ["DISCORD_TOKEN_BOT1"]

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

device = "cuda" if torch.cuda.is_available() else "cpu"

def text_between(string, start, end):
    start_index = string.find(start)
    if start_index != -1:
        start_index += len(start)
        end_index = string.find(end, start_index)
        if end_index == -1:
            end_index = len(string)
        return string[start_index:end_index]

def is_image(filename):
    try:
        with Image.open(filename) as im:
            return True
    except:
        return False

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if 'clear' in message.content:
        await message.channel.purge()

    # Models only requiring text
    if message.content:
        # Stable-diffusion
        if message.content.startswith('draw: '):
            await message.add_reaction('⏳')

            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=512'
            pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-4", torch_dtype=torch.float16).to(device)

            prompt = message.content.replace("draw: ", "")

            filename = f"{prompt}.jpg"
            pipe(prompt).images[0].save(filename)
            # pipe(prompt).images[0].save("image1.jpg")
            # pipe(prompt).images[1].save("image2.jpg")
            # pipe(prompt).images[2].save("image3.jpg")
            
            # image1 = Image.open("image1.jpg")
            # image2 = Image.open("image2.jpg")
            # image3 = Image.open("image3.jpg")

            # # Resize all images to the same size
            # image1 = image1.resize((200, 200))
            # image2 = image2.resize((200, 200))
            # image3 = image3.resize((200, 200))

            # # Create a new blank image for the grid
            # grid_size = (2, 2)
            # grid_image = Image.new('RGB', (grid_size[0] * image1.width, grid_size[1] * image1.height))

            # # Paste each image onto the grid
            # grid_image.paste(image1, (0, 0))
            # grid_image.paste(image2, (image1.width, 0))
            # grid_image.paste(image3, (0, image1.height))

            # filename = f"{prompt}_grid.jpg"
            # grid_image.save(filename)

            # image1.close()
            # image2.close()
            # image3.close()
            # os.remove(f"image1.jpg")
            # os.remove(f"image2.jpg")
            # os.remove(f"image3.jpg")
            
            with open(filename, 'rb') as f:
                try:
                    file = discord.File(f)
                except Exception as e:
                    print(e)
            if is_image(filename):
                print("This is an image")
                await message.remove_reaction('⏳', client.user)
                await message.add_reaction('✅')
                await message.channel.send(prompt, file=file)
            else:
                print("This is not an image")
                await message.remove_reaction('⏳', client.user)
                await message.add_reaction('❌')
            
            f.close()
            os.remove(filename)

    # Models requiring files as input
    elif message.attachments:
        print(message.content)
        for attachment in message.attachments:
            # Download and write file as binary
            r = requests.get(attachment.url)
            with open(attachment.filename, 'wb') as f:
                f.write(r.content)

            await message.add_reaction('⏳')

            try:
                # Whisper
                if attachment.filename.endswith(('mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm')):
                    muligheder = ['transcribe', 'detect']
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
                    if not any(string2 in message.content for string2 in muligheder):
                        text += "I'm not sure what you want me to do?"

                # CLIP
                elif attachment.filename.endswith(('.png', '.jpg', '.webp')) and message.content and message.content.startswith('guess: '):
                    model_name = "ViT-B/32"
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
                        probas.append(np.format_float_positional(item * 100, precision=2) + "%")

                    text_list = []
                    for i in range(len(probas)):
                        text_list.append(f"{possibilities[i]}: {probas[i]}")
                    list_sorted = sorted(text_list, key=lambda x: x.split(": ")[-1][:-1], reverse=True)
                    text = "\n".join(list_sorted)
                
                # Point-E
                # elif

                else:
                    await message.remove_reaction('⏳', client.user)
                    await message.add_reaction('❌')
                    return
            except Exception as e:
                print(e)
                os.remove(attachment.filename)
                await message.remove_reaction('⏳', client.user)
                await message.add_reaction('❌')
                return

            # Delete the file
            os.remove(attachment.filename)
            await message.remove_reaction('⏳', client.user)

            try:
                if text:
                    if len(text) >= 2000:
                        send_queue = textwrap.wrap(text, width=2000)
                        for i, string in enumerate(send_queue):
                            string = string.replace("% ", "%\n")
                            if i == 0:
                                await message.channel.send(string + "`")
                            else:
                                await message.channel.send("`" + string + "`")
                    else:
                        await message.channel.send(text)
                    await message.add_reaction('✅')
            except UnboundLocalError:
                pass
            except discord.errors.HTTPException:
                await message.channel.send("Reached character limit of 2000.")
                await message.add_reaction('⚠️')
                
                



client.run(discord_token)
