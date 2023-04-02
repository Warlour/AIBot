# Processing
from PIL import Image
import os

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
            pass
        return True
    except:
        return False

def seperate_string(input_string:str) -> list[str]:
    if len(input_string) <= 1998:
        return [input_string]
    else:
        return [input_string[i:i+1998] for i in range(0, len(input_string), 1998)]

def find_image_paths(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def create_gif(image_paths, gif_path, duration = 100) -> str:
    images = []
    for path in image_paths:
        image = Image.open(path)
        images.append(image)
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    return gif_path