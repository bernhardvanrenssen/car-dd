import os
import shutil
import re

def select_key_frames_and_filter_images_txt(
    input_dir,
    output_dir,
    step=10,
    images_txt_path=None,
    images_txt_out=None
):
    """
    Copies every 'step'-th image from input_dir to output_dir,
    and also filters the specified images.txt to keep only those lines
    for the selected frames.

    :param input_dir: Directory containing extracted frames (e.g. "./data/images")
    :param output_dir: Destination directory for selected key frames (e.g. "./data/images_key")
    :param step: e.g. 10 to select every 10th frame
    :param images_txt_path: Path to the original COLMAP images.txt file
    :param images_txt_out: Path to write the filtered images_key.txt file
    """

    # 1) Create output_dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 2) Regex to match frames named like frame_00001.png
    pattern = re.compile(r"^frame_(\d+)\.png$")
    
    # 3) Collect all frames in input_dir
    all_files = []
    for fname in os.listdir(input_dir):
        match = pattern.match(fname)
        if match:
            frame_num = int(match.group(1))
            all_files.append((frame_num, fname))
    
    # Sort by frame number
    all_files.sort(key=lambda x: x[0])
    
    # 4) Select every 'step'-th file
    selected = all_files[::step]
    
    # 5) Copy selected frames to output_dir
    for _, fname in selected:
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        shutil.copy(src, dst)
    
    print(f"Copied {len(selected)} files to '{output_dir}'.")

    # 6) If images_txt_path is given, also filter images.txt
    if images_txt_path and images_txt_out:
        # Build a set of selected filenames
        selected_filenames = {fname for (_, fname) in selected}
        
        # Read images.txt, write only lines that correspond to selected frames
        with open(images_txt_path, 'r') as fin, open(images_txt_out, 'w') as fout:
            for line in fin:
                if line.startswith('#') or not line.strip():
                    # Keep comment/blank lines as-is (optional)
                    fout.write(line)
                    continue
                parts = line.strip().split()
                # lines in images.txt typically: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME
                # so we expect exactly 10 tokens
                if len(parts) == 10:
                    image_name = parts[-1]  # e.g. frame_00001.png
                    if image_name in selected_filenames:
                        fout.write(line + "\n")
                else:
                    # If there's an unexpected format, you could handle it or ignore
                    pass
        
        print(f"Filtered images.txt -> '{images_txt_out}' with {len(selected_filenames)} entries.")


if __name__ == "__main__":
    # Example usage:
    # Directories
    input_directory = "./data/images"
    output_directory = "./data/images_key"

    # The COLMAP images.txt input and output
    colmap_images_txt = "./data/images.txt"   # or wherever your images.txt is
    colmap_images_txt_out = "./data/images_key.txt"

    # Step size (every 10th frame)
    step_size = 10
    
    select_key_frames_and_filter_images_txt(
        input_dir=input_directory,
        output_dir=output_directory,
        step=step_size,
        images_txt_path=colmap_images_txt,
        images_txt_out=colmap_images_txt_out
    )
