import numpy as np
from scipy.spatial.transform import Rotation as R

def load_similarity_transformation(file_path):
    """
    Loads a 4x4 similarity transformation matrix from a text file.
    """
    T = np.loadtxt(file_path)
    return T

def update_camera_pose(qw, qx, qy, qz, tx, ty, tz, T_sim):
    """
    Updates a single camera pose given by quaternion (qw, qx, qy, qz) and translation (tx, ty, tz)
    using the similarity transformation T_sim.
    
    Returns:
      new_qw, new_qx, new_qy, new_qz, new_tx, new_ty, new_tz
    """
    # Convert quaternion to rotation matrix.
    # Note: SciPy's Rotation.from_quat expects quaternion in the order [qx, qy, qz, qw]
    rot_cam = R.from_quat([qx, qy, qz, qw])
    R_cam = rot_cam.as_matrix()
    t = np.array([tx, ty, tz])
    # Compute camera center: C = -R_cam^T * t
    C = -R_cam.T @ t
    # Extract scale, rotation, translation from T_sim.
    # T_sim = [ s * R_sim, t_sim ]
    sR = T_sim[:3, :3]
    t_sim = T_sim[:3, 3]
    # Compute scale as the norm of the first column of sR.
    s = np.linalg.norm(sR[:, 0])
    R_sim = sR / s
    # Transform camera center: C' = s * R_sim * C + t_sim
    C_new = s * (R_sim @ C) + t_sim
    # Update rotation: new rotation R_new = R_sim * R_cam
    R_new = R_sim @ R_cam
    # New translation: t_new = -R_new * C_new
    t_new = -R_new @ C_new
    # Convert R_new back to quaternion.
    new_quat = R.from_matrix(R_new).as_quat()  # returns [qx, qy, qz, qw]
    # Rearrange to output: qw, qx, qy, qz
    return new_quat[3], new_quat[0], new_quat[1], new_quat[2], t_new[0], t_new[1], t_new[2]

def process_camera_file(input_file, output_file, T_sim):
    """
    Reads the input camera file (e.g. images_key.txt) and writes a new file with updated camera poses.
    """
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            line = line.strip()
            # Copy comment or empty lines unchanged.
            if line.startswith("#") or len(line) == 0:
                fout.write(line + "\n")
                continue
            tokens = line.split()
            # Expect at least 9 tokens: IMAGE_ID, qw, qx, qy, qz, tx, ty, tz, CAMERA_ID, and then NAME.
            if len(tokens) < 9:
                fout.write(line + "\n")
                continue
            image_id = tokens[0]
            qw = float(tokens[1])
            qx = float(tokens[2])
            qy = float(tokens[3])
            qz = float(tokens[4])
            tx = float(tokens[5])
            ty = float(tokens[6])
            tz = float(tokens[7])
            camera_id = tokens[8]
            name = " ".join(tokens[9:]) if len(tokens) > 9 else ""
            
            new_qw, new_qx, new_qy, new_qz, new_tx, new_ty, new_tz = update_camera_pose(qw, qx, qy, qz, tx, ty, tz, T_sim)
            # Write updated line.
            fout.write(f"{image_id} {new_qw:.8f} {new_qx:.8f} {new_qy:.8f} {new_qz:.8f} "
                       f"{new_tx:.8f} {new_ty:.8f} {new_tz:.8f} {camera_id} {name}\n")

def main():
    # Load the similarity transformation from the file.
    T_sim = load_similarity_transformation("similarity_transformation.txt")
    print("Loaded similarity transformation:")
    print(T_sim)
    
    input_file = "./data/images_key.txt"      # Your original camera file.
    output_file = "images_key_aligned.txt"
    process_camera_file(input_file, output_file, T_sim)
    print(f"Saved updated camera poses to {output_file}")

if __name__ == "__main__":
    main()
