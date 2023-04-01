# Processing
from PIL import Image

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