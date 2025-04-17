#!/usr/bin/env python3
"""
Preprocess HD Images by Removing the Background.

This script uses the rembg library to remove the background from images.
It takes an input folder and an output folder as command-line arguments.
All PNG, JPG, and JPEG images in the input folder will be processed and saved
in the output folder with the same filename. The output images will have transparent
backgrounds where the original background was removed.

Usage:
    python preprocess.py <input_folder> <output_folder>
"""

import os
import sys
from rembg import remove
from PIL import Image
import io

def process_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all files in the input folder.
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            try:
                with open(input_path, 'rb') as i:
                    input_data = i.read()
                # Remove the background using rembg.
                output_data = remove(input_data)
                # Open the resulting image as a PIL image.
                output_image = Image.open(io.BytesIO(output_data)).convert("RGBA")
                
                # Optionally, you can further process the image (crop, resize, etc.)
                # For now, we simply save the output image.
                output_image.save(output_path)
                print(f"Processed {filename} -> {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <input_folder> <output_folder>")
        sys.exit(1)
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    process_images(input_folder, output_folder)
