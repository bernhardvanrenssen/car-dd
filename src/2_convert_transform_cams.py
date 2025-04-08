#!/usr/bin/env python3
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
import argparse
import os

# ----------------------------
# Point Cloud Processing Functions
# ----------------------------
def detect_ground_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                               ransac_n=ransac_n,
                                               num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    print(f"Detected plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    return plane_model, inliers

def compute_rotation_to_align(normal, target_normal=np.array([0, 1, 0])):
    normal = normal / np.linalg.norm(normal)
    target_normal = target_normal / np.linalg.norm(target_normal)
    v = np.cross(normal, target_normal)
    norm_v = np.linalg.norm(v)
    dot_val = np.dot(normal, target_normal)
    if norm_v < 1e-6:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R_matrix = np.eye(3) + vx + vx @ vx * ((1 - dot_val) / (norm_v ** 2))
    return R_matrix

def align_point_cloud(pcd, plane_model):
    a, b, c, d = plane_model
    normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    R_matrix = compute_rotation_to_align(normal, target_normal=np.array([0, 1, 0]))
    pcd.rotate(R_matrix, center=(0, 0, 0))
    return R_matrix

def create_aligned_plane_mesh(y_ground, size=10, thickness=0.01):
    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=thickness)
    plane_mesh.translate(-plane_mesh.get_center())
    R_matrix = o3d.geometry.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
    plane_mesh.rotate(R_matrix, center=(0, 0, 0))
    plane_mesh.translate(np.array([0, y_ground, 0]))
    plane_mesh.paint_uniform_color([0.8, 0.8, 0.8])
    return plane_mesh

def segment_vehicle_aligned(pcd, y_ground, max_height=2.5):
    points = np.asarray(pcd.points)
    indices = np.where((points[:, 1] >= (y_ground - max_height)) & (points[:, 1] <= y_ground))[0]
    vehicle_pcd = pcd.select_by_index(indices)
    print(f"Segmented vehicle: kept {indices.shape[0]} points with y in [{y_ground - max_height:.2f}, {y_ground:.2f}]")
    return vehicle_pcd

def filter_horizontal(pcd, radius=1.0):
    points = np.asarray(pcd.points)
    center_x = np.median(points[:, 0])
    center_z = np.median(points[:, 2])
    print(f"Horizontal center (x,z): ({center_x:.4f}, {center_z:.4f})")
    horizontal_dist = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 2] - center_z)**2)
    indices = np.where(horizontal_dist <= radius)[0]
    filtered_pcd = pcd.select_by_index(indices)
    print(f"After horizontal filtering: kept {indices.shape[0]} points within {radius} m")
    return filtered_pcd

def umeyama_alignment(source, target):
    mu_source = np.mean(source, axis=0)
    mu_target = np.mean(target, axis=0)
    src_centered = source - mu_source
    tgt_centered = target - mu_target
    cov = src_centered.T @ tgt_centered / source.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    R_matrix = Vt.T @ U.T
    if np.linalg.det(R_matrix) < 0:
        Vt[2, :] *= -1
        R_matrix = Vt.T @ U.T
    var_src = np.sum(np.linalg.norm(src_centered, axis=1)**2) / source.shape[0]
    scale = np.trace(np.diag(D)) / var_src
    t = mu_target - scale * R_matrix @ mu_source
    T = np.eye(4)
    T[:3, :3] = scale * R_matrix
    T[:3, 3] = t
    return T, scale, R_matrix, t

def sample_ground_points(ground_cloud, num_points=1000):
    points = np.asarray(ground_cloud.points)
    if points.shape[0] > num_points:
        indices = np.random.choice(points.shape[0], num_points, replace=False)
        return points[indices]
    else:
        return points

def compute_similarity_transformation_from_ground(source_ground, target_ground_cloud):
    target_points = np.asarray(target_ground_cloud.points)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_points)
    distances, indices = nbrs.kneighbors(source_ground)
    target_corr = target_points[indices.flatten()]
    T, scale, R_matrix, t = umeyama_alignment(source_ground, target_corr)
    return T, scale, R_matrix, t

def process_point_cloud(input_file, output_prefix, max_height=2.5, horizontal_radius=4.0, color=[0, 0, 1], adjust_ground=True):
    print(f"\nProcessing {input_file} ...")
    pcd = o3d.io.read_point_cloud(input_file)
    plane_model, inliers = detect_ground_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000)
    ground_cloud = pcd.select_by_index(inliers)
    ground_cloud.paint_uniform_color([1, 0, 0])
    R_align = align_point_cloud(pcd, plane_model)
    print("Applied rotation matrix to align ground:")
    print(R_align)
    ground_cloud.rotate(R_align, center=(0, 0, 0))
    rotated_ground_points = np.asarray(ground_cloud.points)
    y_ground_aligned = np.median(rotated_ground_points[:, 1])
    print(f"Aligned ground level (median y): {y_ground_aligned:.4f}")
    vehicle_pcd = segment_vehicle_aligned(pcd, y_ground_aligned, max_height=max_height)
    vehicle_filtered = filter_horizontal(vehicle_pcd, radius=horizontal_radius)
    vehicle_filtered.paint_uniform_color(color)
    return pcd, vehicle_filtered, ground_cloud, y_ground_aligned, R_align

def process_point_cloud_pipeline(file_ref, file_colmap, max_height, horizontal_radius):
    _, vehicle_ref, ground_ref, ground_level_ref, _ = process_point_cloud(
        file_ref, "ecosport_kiri",
        max_height=max_height,
        horizontal_radius=horizontal_radius,
        color=[0, 1, 1],
        adjust_ground=True)
    aligned_colmap, vehicle_colmap, ground_colmap, ground_level_colmap, R_align_colmap = process_point_cloud(
        file_colmap, "points3D",
        max_height=max_height,
        horizontal_radius=horizontal_radius,
        color=[1, 1, 1],
        adjust_ground=True)
    print(f"Common ground level (from ecosport_kiri): {ground_level_ref:.4f}")
    source_ground = sample_ground_points(ground_colmap, num_points=1000)
    target_ground = sample_ground_points(ground_ref, num_points=1000)
    T_sim, scale_sim, R_sim, t_sim = compute_similarity_transformation_from_ground(source_ground, ground_ref)
    print("Computed similarity transformation (from COLMAP to ecosport_kiri):")
    print(T_sim)
    print(f"Scale: {scale_sim:.4f}")
    print("Rotation:")
    print(R_sim)
    print("Translation:")
    print(t_sim)
    aligned_colmap.transform(T_sim)
    vehicle_colmap.transform(T_sim)
    o3d.io.write_point_cloud("points3D_aligned_filtered_common.ply", vehicle_colmap)
    o3d.io.write_point_cloud("ecosport_kiri_aligned_filtered_common.ply", vehicle_ref)
    print("Exported points3D_aligned_filtered_common.ply and ecosport_kiri_aligned_filtered_common.ply")
    np.savetxt("similarity_transformation.txt", T_sim, fmt="%.6f")
    print("Saved similarity transformation to similarity_transformation.txt")
    return T_sim, R_align_colmap

# ----------------------------
# Automatic Vertical Offset Computation
# ----------------------------
def compute_vertical_offset(original_path, aligned_path):
    if not os.path.exists(original_path):
        print(f"Original file '{original_path}' not found.")
        return 0.0
    if not os.path.exists(aligned_path):
        print(f"Aligned file '{aligned_path}' not found.")
        return 0.0
    orig_cloud = o3d.io.read_point_cloud(original_path)
    aligned_cloud = o3d.io.read_point_cloud(aligned_path)
    bbox_orig = orig_cloud.get_axis_aligned_bounding_box()
    bbox_aligned = aligned_cloud.get_axis_aligned_bounding_box()
    center_orig = bbox_orig.get_center()
    center_aligned = bbox_aligned.get_center()
    vertical_offset = center_orig[1] - center_aligned[1]
    print(f"Computed automatic vertical offset: {vertical_offset:.4f}")
    return vertical_offset

# ----------------------------
# COLMAP Camera Center Conversion Functions
# ----------------------------
def convert_camera_center(qw, qx, qy, qz, tx, ty, tz):
    rot = R.from_quat([qx, qy, qz, qw])
    R_cam = rot.as_matrix()
    t = np.array([tx, ty, tz])
    C = -R_cam.T @ t
    return C

def process_file_camera_centers(input_file, output_file, transform_matrix=None, ignore_scale=False):
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        skip_next = False
        for line in fin:
            line = line.strip()
            if line.startswith("#") or not line:
                fout.write(line + "\n")
                continue
            tokens = line.split()
            if skip_next:
                skip_next = False
                continue
            if len(tokens) < 10:
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
            name = " ".join(tokens[9:])
            C = convert_camera_center(qw, qx, qy, qz, tx, ty, tz)
            if transform_matrix is not None:
                if ignore_scale:
                    s = np.cbrt(np.linalg.det(transform_matrix[:3, :3]))
                    R_no_scale = transform_matrix[:3, :3] / s
                    t_sim = transform_matrix[:3, 3]
                    C = R_no_scale @ C + t_sim
                else:
                    C_hom = np.array([C[0], C[1], C[2], 1.0])
                    C = (transform_matrix @ C_hom)[:3]
            fout.write(f"{image_id} {C[0]:.8f} {C[1]:.8f} {C[2]:.8f} {camera_id} {name}\n")
            skip_next = True

def compute_mean_camera_center(input_file):
    centers = []
    with open(input_file, 'r') as fin:
        skip_next = False
        for line in fin:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            tokens = line.split()
            if skip_next:
                skip_next = False
                continue
            if len(tokens) < 10:
                continue
            qw = float(tokens[1])
            qx = float(tokens[2])
            qy = float(tokens[3])
            qz = float(tokens[4])
            tx = float(tokens[5])
            ty = float(tokens[6])
            tz = float(tokens[7])
            C = convert_camera_center(qw, qx, qy, qz, tx, ty, tz)
            centers.append(C)
            skip_next = True
    centers = np.array(centers)
    return np.mean(centers, axis=0)

def make_transformation_about_center(T, center):
    T_center = np.eye(4)
    T_center[:3, 3] = -center
    T_center_inv = np.eye(4)
    T_center_inv[:3, 3] = center
    return T_center_inv @ T @ T_center

# ----------------------------
# Main Function with Argument Parsing
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Combined script for point cloud processing and COLMAP camera center conversion."
    )
    parser.add_argument("--mode", type=str, choices=["pointcloud", "images", "both"], default="both",
                        help="Functionality to run: 'pointcloud', 'images', or 'both'.")
    parser.add_argument("--ref", type=str, default="./data/ecosport_kiri.ply",
                        help="Reference point cloud file (ecosport_kiri).")
    parser.add_argument("--colmap", type=str, default="./data/points3D.ply",
                        help="COLMAP point cloud file.")
    parser.add_argument("--max_height", type=float, default=2.5,
                        help="Max height for vehicle segmentation.")
    parser.add_argument("--horizontal_radius", type=float, default=4.0,
                        help="Horizontal filtering radius (meters).")
    parser.add_argument("--images_input", type=str, default="./data/images.txt",
                        help="Input COLMAP images.txt file.")
    parser.add_argument("--images_unaligned_output", type=str, default="images_converted_unaligned.txt",
                        help="Output file for original camera centers (unaligned).")
    parser.add_argument("--images_aligned_output", type=str, default="images_converted_aligned.txt",
                        help="Output file for camera centers after applying transformation.")
    parser.add_argument("--ignore_scale", action="store_true",
                        help="If set, ignore the scale component when applying transformation to camera centers.")
    parser.add_argument("--vertical_offset", type=float, default=0.0,
                        help="Manual vertical offset to add (overrides auto if non-zero).")
    parser.add_argument("--auto_offset", action="store_true",
                        help="If set, compute the vertical offset automatically by comparing the original ecosport_kiri.ply bounding box with the processed one.")
    
    args = parser.parse_args()
    
    T_sim = None
    R_align_colmap = None
    if args.mode in ["pointcloud", "both"]:
        print("Running point cloud processing pipeline...")
        T_sim, R_align_colmap = process_point_cloud_pipeline(args.ref, args.colmap, args.max_height, args.horizontal_radius)
    
    if args.mode in ["images", "both"]:
        print("Processing unaligned camera centers (original)...")
        process_file_camera_centers(args.images_input, args.images_unaligned_output, transform_matrix=None)
        
        if T_sim is not None and R_align_colmap is not None:
            T_align = np.eye(4)
            T_align[:3, :3] = R_align_colmap
            T_total = T_sim @ T_align
            
            # Apply manual vertical offset if provided
            if args.vertical_offset != 0.0:
                print(f"Applying manual vertical offset: {args.vertical_offset}")
                T_offset = np.eye(4)
                T_offset[:3, 3] = [0, args.vertical_offset, 0]
                T_total = T_offset @ T_total
            elif args.auto_offset:
                # First, ensure the reference aligned cloud file exists.
                ref_aligned_path = "ecosport_kiri_aligned_filtered_common.ply"
                auto_offset = compute_vertical_offset(args.ref, ref_aligned_path)
                T_offset = np.eye(4)
                T_offset[:3, 3] = [0, auto_offset, 0]
                print(f"Applying auto-computed vertical offset: {auto_offset:.4f}")
                T_total = T_offset @ T_total
            else:
                print("No vertical offset applied.")
            
            mean_center = compute_mean_camera_center(args.images_input)
            print("Mean camera center (unaligned):", mean_center)
            T_total_adjusted = make_transformation_about_center(T_total, mean_center)
            print("Using adjusted total transformation (rotated about center):")
            print(T_total_adjusted)
            print("Processing aligned camera centers (applying adjusted total transformation)...")
            process_file_camera_centers(args.images_input, args.images_aligned_output, transform_matrix=T_total_adjusted, ignore_scale=args.ignore_scale)
        else:
            print("No similarity transformation computed; only unaligned camera centers were generated.")

if __name__ == "__main__":
    main()
