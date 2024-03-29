# Table of Contents
1. [Introduction](#introduction)
2. [Product Description](#product-description)
3. [Why I made the program](#why-i-made-the-program)
4. [What it's useful for](#what-its-useful-for)
5. [Examples](#examples)
    * [OpenJourney](#openjourney)
        * [Example 1](#example-1)
        * [Example 2](#example-2)
6. [Installation](#installation)
    * [Troubleshoot](#troubleshoot)

# Introduction
This program is a Discord bot that utilizes machine learning models to generate images and transcribe audio files for users. The program is written in Python and uses libraries such as Discord.py, PIL, and PyTorch.

# Product description
The bot responds to specific user messages and generates content using machine learning models. Users can provide an image to the bot to run through a CLIP model to guess the image's content. Additionally, users can provide an audio file to transcribe using the Whisper model, and a Stable Diffusion model can generate an image based on user text and image prompts. The bot utilizes the host's GPU (if supported) or CPU, which is also why only one message can be processed at once. Removing this limit would cause performance issues and make the AI models 'give up' due to 'CUDA running out of memory'.

# Why I made the program
I made this program to explore the capabilities of machine learning models and to build a fun bot for the Discord community. By combining different models, users can get more creative results, and the bot can act as a tool for artists or content creators who need inspiration or assistance in generating content.

# What it's useful for
This bot can be used by content creators, artists, and Discord community members to generate images and transcribe audio files. The program can be used as a tool to help users generate ideas or create new content. Additionally, this bot can be used as a learning tool for those interested in machine learning and its applications in creative fields.

# Examples
## OpenJourney
The current primary model used for image generation.
### Example 1
In the following image I used the `/openjourneywithincrease` command.
Specifically the command was:  
`/openjourneywithincrease` `prompt: (((simplistic))), concept art, stylized, splash art, symmetrical, illumination lighting, neural network design,single logo, centered, symbol, shaded, dark shadows, dynamic lighting, watercolor paint, rough paper texture, dark background, darkmode` `increase_guidance_by: 1` `guidance_start: 10` `count: 10` `steps: 50` `creategif: true`

![Orange Logo](https://github.com/Warlour/AIBot/blob/assets/output1.png?raw=true)

The program then creates a gif from the generated images and sends in a separate message:

![Orange Logo GIF](https://github.com/Warlour/AIBot/blob/assets/output1.gif?raw=true)

### Example 2
`/openjourneywithincrease` `prompt: cute isometric cyberpunk bedroom, cutaway box, futuristic, highly detailed, made with blender --v 4` `increase_guidance_by: 1` `guidance_start: 7` `count: 10` `seed: 17168` `creategif: true`

![Isometric cyberpunk bedroom](https://github.com/Warlour/AIBot/blob/assets/output2.png?raw=true)

The program then creates a gif from the generated images and sends in a separate message:

![Isometric cyberpunk bedroom GIF](https://github.com/Warlour/AIBot/blob/assets/output2.gif?raw=true)

# Installation

1. After you have cloned the repository with `git clone https://github.com/Warlour/AIBot.git`, you need to create a folder called `models` inside the new AIBot folder.

2. Use the following git command: `git lfs install`

3. Open a terminal in the "models" folder and install the following repositories (all repos combined take up 124 GB, only download the ones you need!):  
    ​	`git clone https://huggingface.co/prompthero/openjourney` (Mostly used)  
    ​	`git clone https://huggingface.co/CompVis/stable-diffusion-v1-4` (Outdated version, but still works)  
    ​	`git clone https://huggingface.co/PygmalionAI/pygmalion-6b` (Not yet implemented)  

4. Go back to the AIBot folder and run the cmd file: AIBot.cmd - or run the command python3.9 src/command.py from the AIBot folder (cannot be run from the src folder)

## Troubleshoot:  
### No suitable Python runtime found (Make sure to use py -3.9):  
1. Install Python 3.9 from Microsoft Store, (if you're on Linux I'm sure you know how to do this)

### Missing libraries:  
Run the following: `py -3.9 -m pip install discord os requests openai torch diffusers transformers whisper validators pytube clip numpy`

### Make sure you set Discord Token from your <ins>own</ins> bot:  
1. Press `Win+R`, and type the following: `rundll32.exe sysdm.cpl,EditEnvironmentVariables` (this opens Environment Variables)  
2. Click New and insert  
	Variable name as `DISCORD_TOKEN_BOT1`  
        Variable value as your bot token and then click OK.
