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
            return True
    except:
        return False
    
import subprocess
import re

def ytdownload(url: str) -> str:
    result = subprocess.run(["powershell", "-Command", f"youtube-dl.exe {url}"], capture_output=True)
    match = re.search(r'"(.+)"', str(result))
    if match:
        return match.group(1)
    

def seperate_string(input_string:str) -> list[str]:
    if len(input_string) <= 1998:
        return [input_string]
    else:
        return [input_string[i:i+1998] for i in range(0, len(input_string), 1998)]