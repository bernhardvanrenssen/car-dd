import numpy as np

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """ Convert quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix. """
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def convert_colmap_to_bundler(images_txt, output_file):
    """ Convert COLMAP images.txt to CloudCompare-compatible Bundler (.out) format. """
    with open(images_txt, 'r') as f:
        lines = f.readlines()

    # Filter out comments and empty lines
    images = [line.strip() for line in lines if not line.startswith("#") and line.strip()]
    
    num_cameras = len(images) // 2  # Each camera has 2 lines; we need only the first line of each

    if num_cameras == 0:
        print("Error: No camera data found in images.txt")
        return

    with open(output_file, 'w') as f:
        f.write(f"{num_cameras} 0\n")  # No 3D points, just cameras

        for i in range(0, len(images), 2):  # Read every first line (camera params)
            data = images[i].split()
            qw, qx, qy, qz = map(float, data[1:5])
            tx, ty, tz = map(float, data[5:8])

            # Approximate focal length (CloudCompare needs a value, assume ~1000px if unknown)
            focal_length = 1000.0  

            # Convert quaternion to rotation matrix
            R = quaternion_to_rotation_matrix(qw, qx, qy, qz)

            # Write in Bundler format
            f.write(f"{focal_length} 0 0 0\n")  # Focal length, no distortion
            f.write("\n".join(" ".join(map(str, row)) for row in R) + "\n")
            f.write(f"{tx} {ty} {tz}\n")

    print(f"Successfully saved {num_cameras} cameras to {output_file}")

# Example usage:
convert_colmap_to_bundler("images_key.txt", "cameras.out")
