!pip install diffusers transformers
!pip install accelerate
!apt-get install ffmpeg
!pip install opencv-python moviepy
#uncomment if running code on Colab
#from google.colab import drive
#drive.mount('/content/drive')

!apt-get update
!apt-get install imagemagick

!convert -version

import os
os.environ['IMAGEMAGICK_BINARY'] = '/usr/bin/convert'

!apt-get update
!apt-get install imagemagick
!convert -version

!which convert

import os
os.environ['IMAGEMAGICK_BINARY'] = '/usr/bin/convert'  # Adjust this path based on the output of `which convert`

import moviepy.config as mpy_config

mpy_config.change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

!cat /etc/ImageMagick-6/policy.xml

!sudo sed -i 's/<policy domain="path" rights="none" pattern="@\*"/<!--<policy domain="path" rights="none" pattern="@\*"/' /etc/ImageMagick-6/policy.xml
!sudo sed -i 's/<\/policymap>/<\/policymap>-->/' /etc/ImageMagick-6/policy.xml

!cat /etc/ImageMagick-6/policy.xml

!pip install --upgrade moviepy

!pip install pandas openpyxl diffusers transformers torch moviepy wand

import pandas as pd
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, vfx
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch
from IPython.display import HTML, display
from base64 import b64encode

def process_excel_and_generate_videos(excel_file_path, gif_mapping, model_id="CompVis/stable-diffusion-v1-4"):
    # Load Excel file
    df = pd.read_excel(excel_file_path)

    # Load the Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Verify the font installation
    import matplotlib.font_manager as fm
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    liberation_font = [font for font in font_list if 'LiberationSerif-Regular' in font]
    if not liberation_font:
        raise FileNotFoundError("Font 'LiberationSerif-Regular' not found on the system.")

    font_path = liberation_font[0]  # Get the first match

    # Process each row in the DataFrame
    for index, row in df.iterrows():
        if row['Parameters'] == 1:
            machine_no = row['machine no.']
            prompt = row['Prompts']
            text = row['text']
            first_three_letters = machine_no[:3]

            if first_three_letters in gif_mapping:
                gif_path = gif_mapping[first_three_letters]

                # Generate background image based on the prompt
                with torch.autocast("cuda"):
                    image = pipe(prompt).images[0]
                background_path = f"/content/generated_background_{index}.png"
                image.save(background_path)

                # Read the generated background image
                background_image = Image.open(background_path)
                background_image = np.array(background_image)

                # Function to remove black background and add new background
                def remove_black_background_and_add_new(frame):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)
                    bg_resized = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
                    fg = cv2.bitwise_and(frame, frame, mask=mask)
                    bg = cv2.bitwise_and(bg_resized, bg_resized, mask=mask_inv)
                    combined = cv2.add(fg, bg)
                    return combined

                # Read the .mov file
                video = VideoFileClip(gif_path)

                # Apply the background removal function
                new_video = video.fl_image(lambda frame: remove_black_background_and_add_new(frame))

                # Get the size of the video
                video_width, video_height = new_video.size

                # Calculate the y position for 0.2 times the distance from the bottom
                y_position = video_height * 0.80

                # Create a zoom-out text effect
                def make_textclip(t):
                    fontsize = int(65 * (1 - 0.05 * t))  # Decrease fontsize over time
                    text_clip = TextClip(text, fontsize=fontsize, color='white', font=font_path)
                    text_clip = text_clip.set_position((video_width / 2 - text_clip.size[0] / 2, y_position - text_clip.size[1] / 2)).set_start(t).set_duration(0.1)
                    return text_clip

                # Create text clips with zoom-out effect
                duration = video.duration
                text_clips = [make_textclip(t) for t in np.arange(0, duration, 0.1)]
                composite = CompositeVideoClip([new_video] + text_clips).set_duration(duration)

                # Write the video to a temporary .mov file
                temp_video_path = f'/content/temp_video_{index}.mov'
                output_video_path = f'/content/output_video_{index}.mp4'
                composite.write_videofile(temp_video_path, codec='libx264', fps=video.fps)

                # Convert .mov to .mp4 using ffmpeg
                !ffmpeg -i {temp_video_path} {output_video_path}

                # Display the video
                mp4 = open(output_video_path, 'rb').read()
                data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
                display(HTML(f"""
                <video width=400 controls>
                      <source src="{data_url}" type="video/mp4">
                </video>
                """))

# Example usage
excel_file_path = '/content/drive/MyDrive/MESSAGES.xlsx'
gif_mapping = {
    'HMV': '/content/drive/MyDrive/transparent.mov',
    'MYR': '/content/drive/MyDrive/Nanopix Mayur trnsp.mov',
    'SKV': '/content/drive/MyDrive/Nanopix Shuka trnsp.mov'
}

process_excel_and_generate_videos(excel_file_path, gif_mapping)
