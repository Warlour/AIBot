# AIBot

## Introduction
This program is a Discord bot that utilizes machine learning models to generate images and transcribe audio files for users. The program is written in Python and uses libraries such as Discord.py, PIL, and PyTorch.

## Product Description
The bot responds to specific user messages and generates content using machine learning models. Users can provide an image to the bot to run through a CLIP model to guess the image's content. Additionally, users can provide an audio file to transcribe using the Whisper model, and a Stable Diffusion model can generate an image based on user text and image prompts. The bot utilizes the host's GPU (if supported) or CPU, which is also why only one message can be processed at once. Removing this limit would cause performance issues and make the AI models 'give up' due to 'CUDA running out of memory'.

## Why I Made the Program
I made this program to explore the capabilities of machine learning models and to build a fun bot for the Discord community. By combining different models, users can get more creative results, and the bot can act as a tool for artists or content creators who need inspiration or assistance in generating content.

## What it's Useful for
This bot can be used by content creators, artists, and Discord community members to generate images and transcribe audio files. The program can be used as a tool to help users generate ideas or create new content. Additionally, this bot can be used as a learning tool for those interested in machine learning and its applications in creative fields.