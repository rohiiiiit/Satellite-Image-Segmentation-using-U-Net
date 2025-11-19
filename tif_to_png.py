#for satellite tif's
import tifffile as tiff
from PIL import Image
import numpy as np

import os

input_folder = r'M:/PROJECTS/DL_PROJECT/LATEST_CONTENT/LAB_DATASET/2020'
output_folder = r'M:/PROJECTS/DL_PROJECT/LATEST_CONTENT/LAB_DATASET/temp/2020/test/segment'

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".tif") or file.endswith(".tiff"):
        img_path = os.path.join(input_folder, file)
        
        try:
            img_np = tiff.imread(img_path)

            # Normalize if not 8-bit
            if img_np.dtype != 'uint8':
                img_np = (255.0 * (img_np / img_np.max())).astype('uint8')

            img = Image.fromarray(img_np)
            new_name = os.path.splitext(file)[0] + '.png'
            img.save(os.path.join(output_folder, new_name))

        except Exception as e:
            print(f"Error with {file}: {e}")
'''

import os
from PIL import Image
import tifffile as tiff
import numpy as np

input_folder = r'M:/PROJECTS/DL_PROJECT/LATEST_CONTENT/LAB_DATASET/temp/2024/val/segmented'
output_folder = r'M:/PROJECTS/DL_PROJECT/LATEST_CONTENT/LAB_DATASET/temp/2024/val/segment'
os.makedirs(output_folder, exist_ok=True)

# Get all tiff files and sort them numerically
tiff_files = [f for f in os.listdir(input_folder) if f.endswith(('.tif', '.tiff'))]
tiff_files = sorted(tiff_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

processed = []
failed = []

for file in tiff_files:
    img_path = os.path.join(input_folder, file)
    file_number = ''.join(filter(str.isdigit, file))
    
    try:
        # Try opening with PIL first - this preserves color information for many TIFFs
        try:
            img = Image.open(img_path)
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            new_name = file_number + '.png'
            out_path = os.path.join(output_folder, new_name)
            img.save(out_path)
            processed.append(file_number)
            continue
        except Exception as e:
            print(f"PIL approach failed for {file}, trying tifffile: {e}")
        
        # If PIL fails, try with tifffile
        img_array = tiff.imread(img_path)
        
        # Handle various bit depths and formats
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:  # Normalized [0,1]
                img_array = (img_array * 255).astype(np.uint8)
            elif img_array.dtype == np.uint16:  # 16-bit
                img_array = (img_array / 65535 * 255).astype(np.uint8)
            else:  # Other types
                img_array = ((img_array - img_array.min()) * 255 / 
                           (img_array.max() - img_array.min() + 1e-8)).astype(np.uint8)
        
        # Ensure RGB format
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack((img_array,)*3, axis=-1)
        elif len(img_array.shape) > 2 and img_array.shape[2] > 3:  # More than RGB
            img_array = img_array[:, :, :3]  # Keep only RGB channels
            
        # Save with PIL from array
        img = Image.fromarray(img_array)
        new_name = file_number + '.png'
        out_path = os.path.join(output_folder, new_name)
        img.save(out_path)
        processed.append(file_number)
            
    except Exception as e:
        print(f"Failed to process {file} (#{file_number}): {e}")
        failed.append(file_number)

print(f"Conversion complete. Successfully processed {len(processed)} images.")
if failed:
    print(f"Failed to process {len(failed)} images: {', '.join(failed)}")

# Check for any missing files in the expected sequence
all_numbers = set(range(min(int(x) for x in processed), max(int(x) for x in processed) + 1))
processed_numbers = set(map(int, processed))
missing = all_numbers - processed_numbers
if missing:
    print(f"Missing numbers in output: {sorted(missing)}") '''