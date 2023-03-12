import os
import discord
import requests

# Processing
from PIL import Image
import numpy as np
import textwrap

# Models and related
import whisper
import torch
import clip

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

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if 'clear' in message.content:
        await message.channel.purge()
    
    if message.attachments:
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
                    model_parameters = "1550 M"
                    model = whisper.load_model(model_name, device=device)
                    # print(
                    #     f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
                    #     f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
                    # )

                    text = ""
                    if 'transcribe' in message.content:
                        prompt = ""
                        if 'transcribe: ' in message.content:
                            prompt = text_between(message.content, "transcribe: ", " detect")
                            print(prompt)
                        result = model.transcribe(attachment.filename, initial_prompt=prompt)
                        text += f'Transcribed {attachment.filename} using the {model_name} model of {model_parameters} parameters: `{result["text"].strip()}`\n\n'

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
                    model, preprocess = clip.load("ViT-B/32", device=device)

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
