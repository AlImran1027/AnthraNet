# from huggingface_hub import login
# login(new_session=False)

from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import datetime
import os
from pathlib import Path
from tqdm import tqdm

# Initialize model
print("Loading RMBG-2.0 model...")
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])

# Use MPS (Apple Silicon) device
device = 'mps'
print("Using device: MPS (Apple Silicon)")

model.to(device)
model.eval()
print("Model loaded successfully!\n")

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def remove_background(input_image_path):
    """
    Remove background from an image using RMBG-2.0 model.
    
    Args:
        input_image_path (str): Path to the input image
    
    Returns:
        PIL.Image: Image object with background removed
    """
    # Load and preprocess image
    image = Image.open(input_image_path).convert('RGB')
    input_images = transform_image(image).unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    
    return image


def process_folder(input_folder, output_folder=None, supported_formats=None):
    """
    Process all images in a folder and save them with background removed.
    
    Args:
        input_folder (str): Path to the folder containing input images
        output_folder (str, optional): Path to save processed images. 
                                       If None, creates a folder with '_no_bg' suffix
        supported_formats (tuple, optional): Tuple of supported image extensions
    """
    if supported_formats is None:
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
    
    # Convert to Path objects
    input_path = Path(input_folder)
    
    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_folder}")
    
    # Create output folder
    if output_folder is None:
        output_path = input_path.parent / f"{input_path.name}_no_bg"
    else:
        output_path = Path(output_folder)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in supported_formats]
    
    if len(image_files) == 0:
        print(f"No images found in {input_folder}")
        print(f"Supported formats: {supported_formats}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Input folder: {input_path}")
    print(f"Output folder: {output_path}")
    print("-" * 80)
    
    # Process each image
    successful = 0
    failed = 0
    failed_files = []
    
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            # Remove background
            result_image = remove_background(str(img_file))
            
            # Save with same filename but as PNG (to preserve transparency)
            output_file = output_path / f"{img_file.stem}.png"
            result_image.save(output_file)
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {img_file.name}: {str(e)}")
            failed += 1
            failed_files.append(img_file.name)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed_files:
        print("\nFailed files:")
        for fname in failed_files:
            print(f"  - {fname}")
    
    print(f"\nProcessed images saved to: {output_path}")
    print("=" * 80)


def process_dataset_with_subfolders(root_folder, bg_remove_folder, supported_formats=None):
    """
    Process all images in subfolders and save them with background removed.
    Creates new folders with '_BGR' suffix and renames images systematically.
    
    Args:
        root_folder (str): Path to root folder containing subfolders (e.g., 'Combined Dataset')
        bg_remove_folder (str): Path to BG_Remove folder where processed images will be saved
        supported_formats (tuple, optional): Tuple of supported image extensions
    
    Example:
        Input:  Combined Dataset/Guava_Anthracnose/image1.jpg
        Output: BG_Remove/Guava_Anthracnose_BGR/guava_anthracnose_1.png
    """
    if supported_formats is None:
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
    
    root_path = Path(root_folder)
    bg_remove_path = Path(bg_remove_folder)
    
    if not root_path.exists():
        raise ValueError(f"Root folder does not exist: {root_folder}")
    
    # Create BG_Remove folder if it doesn't exist
    bg_remove_path.mkdir(parents=True, exist_ok=True)
    
    # Get all subfolders
    subfolders = [f for f in root_path.iterdir() if f.is_dir()]
    
    if len(subfolders) == 0:
        print(f"No subfolders found in {root_folder}")
        return
    
    print("=" * 80)
    print("BATCH BACKGROUND REMOVAL - DATASET PROCESSING")
    print("=" * 80)
    print(f"Root folder: {root_path}")
    print(f"Output folder: {bg_remove_path}")
    print(f"Found {len(subfolders)} subfolders to process")
    print("=" * 80 + "\n")
    
    total_successful = 0
    total_failed = 0
    
    # Process each subfolder
    for subfolder in subfolders:
        folder_name = subfolder.name
        print(f"\n{'─' * 80}")
        print(f"Processing: {folder_name}")
        print(f"{'─' * 80}")
        
        # Create output folder with _BGR suffix
        output_folder_name = f"{folder_name}_BGR"
        output_folder_path = bg_remove_path / output_folder_name
        output_folder_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in subfolder.iterdir() 
                       if f.is_file() and f.suffix.lower() in supported_formats]
        
        if len(image_files) == 0:
            print(f"  No images found in {folder_name}")
            continue
        
        print(f"  Found {len(image_files)} images")
        
        # Create base name from folder name (e.g., 'Guava_Anthracnose' -> 'guava_anthracnose')
        base_name = folder_name.lower().replace(' ', '_')
        
        # Process each image
        successful = 0
        failed = 0
        failed_files = []
        
        for idx, img_file in enumerate(tqdm(image_files, desc=f"  {folder_name}", leave=False), start=1):
            try:
                # Remove background
                result_image = remove_background(str(img_file))
                
                # Create new filename: guava_anthracnose_1.png, guava_anthracnose_2.png, etc.
                new_filename = f"{base_name}_{idx}.png"
                output_file = output_folder_path / new_filename
                result_image.save(output_file)
                successful += 1
                
            except Exception as e:
                print(f"\n  Error processing {img_file.name}: {str(e)}")
                failed += 1
                failed_files.append(img_file.name)
        
        # Folder summary
        print(f"  ✓ Successful: {successful}/{len(image_files)}")
        if failed > 0:
            print(f"  ✗ Failed: {failed}")
            for fname in failed_files:
                print(f"    - {fname}")
        print(f"  Output: {output_folder_path}")
        
        total_successful += successful
        total_failed += failed
    
    # Final summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE - FINAL SUMMARY")
    print("=" * 80)
    print(f"Folders processed: {len(subfolders)}")
    print(f"Total images successful: {total_successful}")
    print(f"Total images failed: {total_failed}")
    print(f"All processed images saved to: {bg_remove_path}")
    print("=" * 80)


# Example usage
if __name__ == "__main__":
    # ========================================================================================
    # METHOD 1: Process entire dataset with subfolders (RECOMMENDED FOR YOUR USE CASE)
    # ========================================================================================
    # This will process all subfolders in 'Combined Dataset' and save to 'BG_Remove'
    # Input:  Combined Dataset/Guava_Anthracnose/*.jpg
    # Output: BG_Remove/Guava_Anthracnose_BGR/guava_anthracnose_1.png, guava_anthracnose_2.png, ...
    
    root_folder = "/Users/alimran/Desktop/CSE465/Dataset"
    bg_remove_folder = "/Users/alimran/Desktop/CSE465"
    
    process_dataset_with_subfolders(root_folder, bg_remove_folder)
    
    
    # ========================================================================================
    # METHOD 2: Process single folder (if you want to process just one folder)
    # ========================================================================================
    # input_folder = "/Users/alimran/Desktop/test"
    # process_folder(input_folder)
    
    
    # ========================================================================================
    # METHOD 3: Process single image (if you want to test on one image)
    # ========================================================================================
    # input_image_path = "/Users/alimran/Desktop/test/Healthy(4).jpg"
    # result_image = remove_background(input_image_path)
    # result_image.save("output_no_bg.png")
    # print("Background removed successfully")