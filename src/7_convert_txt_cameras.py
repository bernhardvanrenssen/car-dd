#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_camera_center(qw, qx, qy, qz, tx, ty, tz):
    """
    Given COLMAP camera pose parameters (quaternion and translation),
    computes the camera center as:
         C = - R_cam^T * t
    COLMAP stores the quaternion as (qw, qx, qy, qz) (in that order).
    SciPy’s Rotation.from_quat expects [qx, qy, qz, qw].
    """
    # Convert quaternion to rotation matrix.
    rot = R.from_quat([qx, qy, qz, qw])
    R_cam = rot.as_matrix()
    t = np.array([tx, ty, tz])
    # Compute camera center
    C = -R_cam.T @ t
    return C

def process_file(input_file, output_file):
    """
    Reads the COLMAP images.txt file, and for each camera (first line of each pair)
    computes the camera center and writes a new file with only the camera parameters.
    
    Output format:
      IMAGE_ID C_x C_y C_z CAMERA_ID NAME
    """
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        skip_next = False  # flag to skip the POINTS2D line
        for line in fin:
            line = line.strip()
            # If line is a comment or empty, copy it.
            if line.startswith("#") or not line:
                fout.write(line + "\n")
                continue
            # Split tokens.
            tokens = line.split()
            # If we already processed a camera line, then the next line is likely the POINTS2D data.
            # COLMAP images.txt has two lines per image: first is the camera pose; second is the 2D observations.
            # We'll skip the second line.
            if skip_next:
                skip_next = False
                continue
            # We expect a valid camera pose line to have at least 10 tokens.
            if len(tokens) < 10:
                continue  # skip if not a valid camera pose line
            # Process the camera pose line.
            image_id = tokens[0]
            qw = float(tokens[1])
            qx = float(tokens[2])
            qy = float(tokens[3])
            qz = float(tokens[4])
            tx = float(tokens[5])
            ty = float(tokens[6])
            tz = float(tokens[7])
            camera_id = tokens[8]
            name = " ".join(tokens[9:])
            # Compute camera center.
            C = convert_camera_center(qw, qx, qy, qz, tx, ty, tz)
            # Write out the result:
            fout.write(f"{image_id} {C[0]:.8f} {C[1]:.8f} {C[2]:.8f} {camera_id} {name}\n")
            # Set flag to skip the next line (the POINTS2D data).
            skip_next = True

def main():
    input_file = "./data/images.txt"          # Your full COLMAP images file.
    output_file = "images_converted.txt"  # New file with computed camera centers.
    process_file(input_file, output_file)
    print(f"Converted camera centers have been saved to {output_file}")

if __name__ == "__main__":
    main()
