#!/usr/bin/env python3
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.transform import Rotation as R
import argparse
import os
import random

DEBUG = True  # Set to True to enable debug prints for the transformation.

# ===================================================
# ORIGINAL FUNCTIONS FOR POINT CLOUD PROCESSING
# ===================================================

def sample_farthest(input_file, output_file, total_samples):
    # 1. Read positions and lines
    cams = []
    for line in open(input_file):
        if line.startswith("#") or not line.strip(): continue
        parts = line.split()
        Cx, Cy, Cz = map(float, parts[5:8])
        cams.append(((Cx, Cy, Cz), line.rstrip()))

    pts = np.array([c[0] for c in cams])
    N   = len(pts)
    if total_samples >= N:
        sel = list(range(N))
    else:
        # 2. Start with a random camera
        sel = [ random.randrange(N) ]
        # 3. Greedily add the farthest next point
        for _ in range(1, total_samples):
            # compute distance from each point to nearest selected
            dists = np.min(
                np.linalg.norm(pts[:,None,:] - pts[sel][None,:,:], axis=2),
                axis=1
            )
            # exclude already selected
            dists[sel] = -1
            nxt = int(np.argmax(dists))
            sel.append(nxt)

    # 4. Write them out
    with open(output_file, "w") as fout:
        fout.write("# Key camera centers via farthest‑point sampling\n")
        for i in sel:
            fout.write(cams[i][1] + "\n")

def sample_passwise_2d_fps(input_file, output_file,
                           total_samples,
                           center_fraction=0.6):
    """
    Reads your aligned images file (with 'Cx Cy Cz ... cluster'),
    splits into the 3 cluster labels, and does a 2D farthest‑point
    sampling on (Cx, Cz) in each pass individually.
    """
    # 1) read in (Cx,Cy,Cz,cluster,line)
    cams = []
    for line in open(input_file):
        if line.startswith("#") or not line.strip(): continue
        parts = line.split()
        Cx, Cy, Cz = map(float, parts[5:8])
        cluster    = int(parts[-1])
        cams.append(((Cx, Cz, cluster), line.rstrip()))
    # separate by pass
    passes = {}
    for (Cx, Cz, cl), line in cams:
        passes.setdefault(cl, []).append((np.array([Cx, Cz]), line))

    # 2) decide how many in each
    mid_n = int(total_samples * center_fraction)
    rem   = total_samples - mid_n
    bot_n = rem // 2
    top_n = rem - bot_n

    # map cluster labels to roles:
    # assume you have labels 0,1,2 but you want to know which is middle
    Ys    = np.array([c[0][1] for c in cams]).reshape(-1, 1)
    km    = KMeans(n_clusters=3, random_state=0).fit(Ys)
    centers = km.cluster_centers_.flatten()
    order   = np.argsort(centers)
    bottom, middle, top = order

    target_counts = {
        bottom: bot_n,
        middle: mid_n,
        top:    top_n
    }

    # 3) do 2D‐FPS in each pass
    selected_lines = []
    for cl, pts in passes.items():
        coords = np.stack([p[0] for p in pts], axis=0)
        lines  = [p[1] for p in pts]
        want   = target_counts.get(cl, 0)
        N      = len(coords)
        if want >= N:
            # take them all
            selected = list(range(N))
        else:
            # fps in 2D
            sel = [ random.randrange(N) ]
            for _ in range(1, want):
                # distance to nearest already selected
                dmat = np.linalg.norm(coords[:,None,:] - coords[sel][None,:,:], axis=2)
                mind = dmat.min(axis=1)
                mind[sel] = -1    # exclude
                nxt = int(np.argmax(mind))
                sel.append(nxt)
            selected = sel

        selected_lines += [ lines[i] for i in selected ]

    # 4) write out exactly total_samples
    with open(output_file, "w") as fout:
        fout.write("# Key cameras via passwise 2D‐FPS\n")
        for line in selected_lines[:total_samples]:
            fout.write(line + "\n")

def sample_weighted_by_pass(input_file, output_file, total_samples, center_frac=0.6):
    """
    - Clusters cameras into 3 passes by height (Y).
    - Allocates `center_frac` of samples to the middle pass,
      and divides the rest equally between top & bottom.
    """
    # 1. Read all lines, parse Y (Cy) and keep full lines
    cams = []
    for line in open(input_file):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        # qw,qx,qy,qz,Cx,Cy,Cz,camId,name,cluster
        Cy = float(parts[6])
        cams.append((Cy, line.rstrip()))

    Ys = np.array([c[0] for c in cams]).reshape(-1,1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(Ys)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()
    # sort clusters by height
    order = np.argsort(centers)
    bottom, middle, top = order

    # 2. Compute how many to draw from each
    center_n = int(total_samples * center_frac)
    other_n  = (total_samples - center_n) // 2
    counts   = {
        middle: center_n,
        bottom: other_n,
        top:    total_samples - center_n - other_n
    }

    # 3. Randomly sample from each cluster
    sampled = []
    for cluster_label, n in counts.items():
        idxs = [i for i, lab in enumerate(labels) if lab == cluster_label]
        sampled += random.sample(idxs, min(n, len(idxs)))

    # 4. Write out exactly `total_samples` lines
    with open(output_file, "w") as fout:
        fout.write("# Key camera centers sampled by pass weights\n")
        for i in sampled[:total_samples]:
            fout.write(cams[i][1] + "\n")

def sample_selective_band(input_file, output_file, total_samples):
    """
    Samples cameras from the middle band (if 3 bands exist) or lower band (if 2 bands exist).
    The selected cameras are spaced as evenly as possible within that band.
    
    Args:
        input_file: Path to the input file containing camera information with cluster labels
        output_file: Path to save the selected cameras
        total_samples: Number of cameras to sample
    """
    # 1. Read the camera information and extract cluster data
    cameras = []
    with open(input_file, 'r') as fin:
        for line in fin:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 11:  # Ensure we have enough parts including the cluster label
                continue
            
            # Extract camera position and cluster
            Cx, Cy, Cz = map(float, parts[5:8])
            cluster = int(parts[-1])
            cameras.append({
                "pos": (Cx, Cy, Cz),
                "cluster": cluster,
                "line": line.rstrip()
            })
    
    if not cameras:
        print("No cameras found in the input file.")
        return
    
    # 2. Identify unique clusters and their counts
    clusters = {}
    for cam in cameras:
        cluster = cam["cluster"]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(cam)
    
    num_clusters = len(clusters)
    print(f"Found {num_clusters} clusters")
    
    # 3. Determine which cluster to use
    if num_clusters >= 3:
        # Use the middle cluster (sort clusters by median Y-coordinate)
        cluster_heights = []
        for cluster_id, cluster_cams in clusters.items():
            median_y = np.median([cam["pos"][1] for cam in cluster_cams])
            cluster_heights.append((cluster_id, median_y))
        
        # Sort by height (Y-coordinate)
        cluster_heights.sort(key=lambda x: x[1])
        
        # Select the middle cluster
        target_cluster = cluster_heights[len(cluster_heights) // 2][0]
        print(f"Using middle cluster {target_cluster}")
    elif num_clusters == 2:
        # Use the lower cluster (with smaller Y-coordinate)
        cluster_heights = []
        for cluster_id, cluster_cams in clusters.items():
            median_y = np.median([cam["pos"][1] for cam in cluster_cams])
            cluster_heights.append((cluster_id, median_y))
        
        # Sort by height and select the lower one
        cluster_heights.sort(key=lambda x: x[1])
        target_cluster = cluster_heights[0][0]
        print(f"Using lower cluster {target_cluster}")
    else:
        # If only one cluster, use it
        target_cluster = list(clusters.keys())[0]
        print(f"Using only available cluster {target_cluster}")
    
    # 4. Get cameras from the selected cluster
    selected_cameras = clusters[target_cluster]
    
    # 5. Calculate spatial distribution for even sampling
    # First compute the 2D positions (X,Z) of cameras in the selected cluster
    positions_2d = np.array([[cam["pos"][0], cam["pos"][2]] for cam in selected_cameras])
    
    # Calculate the center of the distribution
    center = np.mean(positions_2d, axis=0)
    
    # Calculate polar angles relative to the center
    angles = []
    for pos in positions_2d:
        dx = pos[0] - center[0]
        dz = pos[1] - center[1]  # Note: positions_2d[1] is Z coordinate
        angle = np.arctan2(dz, dx)
        angles.append(angle)
    
    # Pair cameras with their angles
    cam_angles = [(cam, angle) for cam, angle in zip(selected_cameras, angles)]
    
    # Sort by angle for even distribution around the center
    cam_angles.sort(key=lambda x: x[1])
    
    # 6. Select evenly spaced cameras based on their angular distribution
    selected = []
    if len(cam_angles) <= total_samples:
        # If we have fewer cameras than requested, use all of them
        selected = [cam for cam, _ in cam_angles]
    else:
        # Calculate evenly spaced indices
        step = len(cam_angles) / total_samples
        indices = [int(i * step) for i in range(total_samples)]
        selected = [cam_angles[i][0] for i in indices]
    
    # 7. Write the selected cameras to the output file
    with open(output_file, "w") as fout:
        fout.write(f"# Key cameras from {'middle' if num_clusters >= 3 else 'lower'} band, evenly spaced\n")
        for cam in selected:
            fout.write(cam["line"] + "\n")
    
    print(f"Selected {len(selected)} cameras from cluster {target_cluster}")
    return

def detect_ground_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                               ransac_n=ransac_n,
                                               num_iterations=num_iterations)
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
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
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
    return vehicle_pcd

def filter_horizontal(pcd, radius=1.0):
    points = np.asarray(pcd.points)
    center_x = np.median(points[:, 0])
    center_z = np.median(points[:, 2])
    horizontal_dist = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 2] - center_z)**2)
    indices = np.where(horizontal_dist <= radius)[0]
    filtered_pcd = pcd.select_by_index(indices)
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
    T[:3, :3] = scale * R_matrix   # s*R
    T[:3, 3] = t                  # translation vector
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
    pcd = o3d.io.read_point_cloud(input_file)
    plane_model, inliers = detect_ground_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000)
    ground_cloud = pcd.select_by_index(inliers)
    ground_cloud.paint_uniform_color([1, 0, 0])
    R_align = align_point_cloud(pcd, plane_model)
    ground_cloud.rotate(R_align, center=(0, 0, 0))
    rotated_ground_points = np.asarray(ground_cloud.points)
    y_ground_aligned = np.median(rotated_ground_points[:, 1])
    vehicle_pcd = segment_vehicle_aligned(pcd, y_ground_aligned, max_height=max_height)
    vehicle_filtered = filter_horizontal(vehicle_pcd, radius=horizontal_radius)
    vehicle_filtered.paint_uniform_color(color)
    return pcd, vehicle_filtered, ground_cloud, y_ground_aligned, R_align

# ===================================================
# ADDED FUNCTIONS FOR CAMERA CENTER PROCESSING
# ===================================================

def convert_camera_center(qw, qx, qy, qz, tx, ty, tz):
    # Convert COLMAP camera parameters into a camera center.
    # Compute the rotation matrix from the quaternion then calculate:
    # C = -R_cam^T * t
    # Finally, flip X and Y to match ecosport_kiri coordinates.
    rot = R.from_quat([qx, qy, qz, qw])
    R_cam = rot.as_matrix()
    t = np.array([tx, ty, tz])
    C = -R_cam.T @ t
    # UNCOMMENT THE FOLLOWING TO FILP THE Y AND X AXIS>>>>
    # C[1] = -C[1]
    # C[0] = -C[0]
    return C

def auto_cluster_y_values(y_vals):
    y_arr = np.array(y_vals).reshape(-1, 1)
    candidate_ks = [2, 3, 4, 5]
    best_k = None
    best_score = -1
    scores = {}
    for k in candidate_ks:
        if len(y_arr) < k:
            continue
        kmeans = KMeans(n_clusters=k, random_state=42).fit(y_arr)
        labels = kmeans.labels_
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(y_arr, labels)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    if 3 in scores and scores[3] >= 0.9 * best_score:
        best_k = 3
    if best_k is None:
        return None, None
    kmeans = KMeans(n_clusters=best_k, random_state=42).fit(y_arr)
    best_labels = kmeans.labels_
    return best_labels, best_k

def process_file_camera_centers(input_file, output_file, transform_matrix=None, ignore_scale=False):
    cameras = []
    with open(input_file, 'r') as fin:
        lines = fin.readlines()
    skip_next = False
    for line in lines:
        line = line.strip()
        if line.startswith("#") or not line:
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
            # Note: The axis-flipping after transformation is disabled.
        cameras.append({
            "image_id": image_id,
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
            "C": C,
            "camera_id": camera_id,
            "name": name,
        })
        skip_next = True
    y_vals = [cam["C"][1] for cam in cameras]
    labels, best_k = auto_cluster_y_values(y_vals)
    if best_k is None:
        labels = [-1] * len(cameras)
    with open(output_file, 'w') as fout:
        fout.write("# image_id qw qx qy qz Cx Cy Cz camera_id name cluster_label\n")
        for cam, label in zip(cameras, labels):
            qw, qx, qy, qz = cam["qw"], cam["qx"], cam["qy"], cam["qz"]
            Cx, Cy, Cz    = cam["C"]
            fout.write(
                f"{cam['image_id']} "
                f"{qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                f"{Cx:.8f} {Cy:.8f} {Cz:.8f} "
                f"{cam['camera_id']} {cam['name']} {label}\n"
            )
    if DEBUG and transform_matrix is not None:
        print("Debug: First 3 camera centers after transformation:")
        for i in range(min(3, len(cameras))):
            print(f"{cameras[i]['image_id']} {cameras[i]['C']}")
    return

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

def sample_key_cameras(input_file, output_file, num_cameras):
    with open(input_file, 'r') as fin:
        lines = [line.rstrip() for line in fin if line and not line.startswith("#")]
    total = len(lines)
    if total == 0:
        return
    step = max(total // num_cameras, 1)
    selected = [lines[i] for i in range(0, total, step)]
    selected = selected[:num_cameras]
    with open(output_file, 'w') as fout:
        fout.write("# Key camera centers sampled evenly\n")
        for line in selected:
            fout.write(line + "\n")
    return

# ===================================================
# MAIN FUNCTION
# ===================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process point clouds to compute a similarity transformation and then export COLMAP camera centers with pass clustering."
    )
    parser.add_argument("--mode", type=str, choices=["pointcloud", "images", "both"], default="both")
    parser.add_argument("--ref", type=str, default="./data/ecosport_kiri.ply")
    parser.add_argument("--colmap", type=str, default="./data/points3D.ply")
    parser.add_argument("--max_height", type=float, default=2.5)
    parser.add_argument("--horizontal_radius", type=float, default=4.0)
    parser.add_argument("--images_input", type=str, default="./data/images.txt")
    parser.add_argument("--images_unaligned_output", type=str, default="images_converted_unaligned.txt")
    parser.add_argument("--images_aligned_output", type=str, default="images_converted.txt")
    parser.add_argument("--ignore_scale", action="store_true")
    parser.add_argument("--vertical_offset", type=float, default=0.0)
    parser.add_argument("--auto_offset", dest="auto_offset", action="store_true")
    parser.add_argument("--no_auto_offset", dest="auto_offset", action="store_false")
    parser.set_defaults(auto_offset=True)
    parser.add_argument("--cameras", type=int, default=30)
    parser.add_argument("--sampling", type=str, choices=["fps", "passwise", "weighted", "selective_band"], default="selective_band")
    args = parser.parse_args()
    
    T_sim = None
    R_align_colmap = None
    
    # Process point clouds
    if args.mode in ["pointcloud", "both"]:
        _, vehicle_ref, ground_ref, ground_level_ref, _ = process_point_cloud(
            args.ref, "ecosport_kiri",
            max_height=args.max_height,
            horizontal_radius=args.horizontal_radius,
            color=[0, 1, 1],
            adjust_ground=True)
        _, vehicle_colmap, ground_colmap, ground_level_colmap, R_align_colmap = process_point_cloud(
            args.colmap, "points3D",
            max_height=args.max_height,
            horizontal_radius=args.horizontal_radius,
            color=[1, 1, 1],
            adjust_ground=True)
        source_ground = sample_ground_points(ground_colmap, num_points=1000)
        target_ground = sample_ground_points(ground_ref, num_points=1000)
        T_sim, scale_sim, R_sim, t_sim = compute_similarity_transformation_from_ground(source_ground, ground_ref)
        if DEBUG:
            print("Debug: Similarity Transformation (T_sim):")
            print(T_sim)
        # Compute full transformation for cameras.
        H = np.eye(4)
        H[:3, :3] = R_align_colmap
        T_full = T_sim @ H
        if DEBUG:
            print("Debug: Full Transformation (T_full):")
            print(T_full)
        # For the point cloud, apply only the similarity transformation (no extra rotation).
        aligned_colmap = vehicle_colmap  # Use processed point cloud
        aligned_colmap.transform(T_sim)
        o3d.io.write_point_cloud("points3D_aligned_filtered_common.ply", aligned_colmap)
        o3d.io.write_point_cloud("ecosport_kiri_aligned_filtered_common.ply", vehicle_ref)
        np.savetxt("similarity_transformation.txt", T_full, fmt="%.6f")
    
    # KEEP THIS FOR NOT FLIPPING
    if args.mode in ["images", "both"]:
        process_file_camera_centers(args.images_input, args.images_unaligned_output, transform_matrix=None)
        if T_sim is not None and R_align_colmap is not None:
            T_total = T_full.copy()
            if args.vertical_offset != 0.0:
                T_offset = np.eye(4)
                T_offset[:3, 3] = [0, -args.vertical_offset, 0]
                T_total = T_offset @ T_total
            elif args.auto_offset:
                auto_offset = 0.0
                T_offset = np.eye(4)
                T_offset[:3, 3] = [0, -auto_offset, 0]
                T_total = T_offset @ T_total
            if DEBUG:
                print("Debug: Using transformation for cameras (T_total):")
                print(T_total)
            process_file_camera_centers(args.images_input, args.images_aligned_output, transform_matrix=T_total, ignore_scale=args.ignore_scale)
        else:
            process_file_camera_centers(args.images_input, args.images_aligned_output, transform_matrix=None)
    
    # Apply sampling method based on user choice
    if args.cameras > 0 and os.path.exists(args.images_aligned_output):
        if args.sampling == "fps":
            sample_farthest(args.images_aligned_output, "key_images.txt", args.cameras)
        elif args.sampling == "passwise":
            sample_passwise_2d_fps(args.images_aligned_output, "key_images.txt", args.cameras)
        elif args.sampling == "weighted":
            sample_weighted_by_pass(args.images_aligned_output, "key_images.txt", args.cameras)
        elif args.sampling == "selective_band":
            sample_selective_band(args.images_aligned_output, "key_images.txt", args.cameras)
        else:
            print(f"Unknown sampling method: {args.sampling}")

if __name__ == "__main__":
    main()