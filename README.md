# AI_NANOPIX

**# Purpose:**
The purpose of this script is to process an Excel file containing machine details and prompts, generate background images using Stable Diffusion based on the provided prompts, overlay the text on pre-defined GIFs corresponding to machine numbers, and produce output videos with specific text effects.
**# Pre-requisites:**
1.	Python Libraries: Ensure the following Python libraries are installed:
    o	pandas
    o	openpyxl
    o	diffusers
    o	transformers
    o	torch
    o	moviepy
    o	wand
    o	PIL
    o	matplotlib
2.	Excel File: An Excel file (MESSAGES.xlsx) containing columns for machine numbers, prompts, text, and parameters.
3.	Pre-defined GIFs: A dictionary mapping the first three letters of machine numbers to corresponding .MOV file paths.
4.	Stable Diffusion Model: Pre-trained Stable Diffusion model available from the model ID "CompVis/stable-diffusion-v1-4".


**# Input:**
**1.	Excel File Columns:**
o	machine no.: Machine number (used to map to a specific GIF).
o	Prompts: Prompt text for generating background images.
o	text: Text to be overlayed on the .MOV file.
o	Parameters: Indicator (1 or 0) to process the row or not. A ‘1’ indicates that the parameter exceeds our threshold and hence, a gif must be created. A ‘0’ indicated that there is no issue and hence, no gifs or alerts are created.
**2.	GIF Mapping:** Dictionary mapping the first three letters of machine numbers to corresponding .MOV file paths.
**Output:**
•	Generated videos (.mp4 files) with the text overlayed on GIFs and a background image created from the prompt.
**Steps:**
1.	Load Excel File: Read the Excel file into a DataFrame.
2.	Load Stable Diffusion Model: Load the pre-trained Stable Diffusion model for image generation.
3.	Verify Font Installation: Ensure the required font (LiberationSerif-Regular) is installed on the system.
4.	Process Each Row in the DataFrame:
o	Check if the Parameters column is set to 1.
o	Extract the first three letters of the machine number to find the corresponding GIF path.
o	Generate a background image using the provided prompt.
o	Combine the background image with the GIF, removing any black background from the GIF.
o	Create a text clip with a zoom-out effect, positioned at 0.25 times the distance from the bottom of the video.
o	Overlay the text clip on the GIF.
o	Save the resulting video as a .mp4 file.
5.	Display the Output Video: Convert the video to a data URL and display it inline.
