import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors

def detect_ground_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    """
    Detects the dominant plane in the point cloud using RANSAC.
    Returns the plane coefficients and the indices of the inlier points.
    """
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                               ransac_n=ransac_n,
                                               num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    print(f"Detected plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    return plane_model, inliers

def compute_rotation_to_align(normal, target_normal=np.array([0, 1, 0])):
    """
    Computes a rotation matrix that rotates the given unit 'normal' to align with 'target_normal'.
    """
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
    R = np.eye(3) + vx + vx @ vx * ((1 - dot_val) / (norm_v ** 2))
    return R

def align_point_cloud(pcd, plane_model):
    """
    Rotates the point cloud so that the ground plane's normal aligns with [0,1,0].
    Returns the rotation matrix.
    """
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    R = compute_rotation_to_align(normal, target_normal=np.array([0, 1, 0]))
    pcd.rotate(R, center=(0, 0, 0))
    return R

def create_aligned_plane_mesh(y_ground, size=10, thickness=0.01):
    """
    Creates a horizontal plane mesh (aligned with Y-up) at the given ground level y_ground.
    """
    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=thickness)
    plane_mesh.translate(-plane_mesh.get_center())
    # Rotate 90° about X so that the front face (originally +z) becomes +y.
    R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
    plane_mesh.rotate(R, center=(0, 0, 0))
    plane_mesh.translate(np.array([0, y_ground, 0]))
    plane_mesh.paint_uniform_color([0.8, 0.8, 0.8])
    return plane_mesh

def segment_vehicle_aligned(pcd, y_ground, max_height=2.5):
    """
    Assumes the point cloud has been rotated so that the ground is level (Y-up).
    For an inverted system, we assume the ground level is y_ground and keep points with Y in:
         [y_ground - max_height, y_ground]
    """
    points = np.asarray(pcd.points)
    indices = np.where((points[:, 1] >= (y_ground - max_height)) & (points[:, 1] <= y_ground))[0]
    vehicle_pcd = pcd.select_by_index(indices)
    print(f"Segmented vehicle: kept {indices.shape[0]} points with y in [{y_ground - max_height:.2f}, {y_ground:.2f}]")
    return vehicle_pcd

def filter_horizontal(pcd, radius=1.0):
    """
    Filters the point cloud by keeping only points within 'radius' meters
    (in the XZ-plane) of the horizontal center (median x and z).
    """
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
    """
    Computes the similarity transformation (rotation, translation, and uniform scale)
    that best aligns the source points to the target points using the Umeyama algorithm.
    Returns:
      T: 4x4 transformation matrix such that target ≈ T * [source; 1]
      scale: uniform scale factor
      R: rotation matrix (3x3)
      t: translation vector (3,)
    """
    mu_source = np.mean(source, axis=0)
    mu_target = np.mean(target, axis=0)
    src_centered = source - mu_source
    tgt_centered = target - mu_target
    cov = src_centered.T @ tgt_centered / source.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    var_src = np.sum(np.linalg.norm(src_centered, axis=1)**2) / source.shape[0]
    scale = np.trace(np.diag(D)) / var_src
    t = mu_target - scale * R @ mu_source
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T, scale, R, t

def sample_ground_points(ground_cloud, num_points=1000):
    """
    Randomly samples up to num_points from the ground_cloud.
    """
    points = np.asarray(ground_cloud.points)
    if points.shape[0] > num_points:
        indices = np.random.choice(points.shape[0], num_points, replace=False)
        return points[indices]
    else:
        return points

def compute_similarity_transformation_from_ground(source_ground, target_ground_cloud):
    """
    For each point in source_ground (numpy array of shape (N,3)), find its nearest neighbor
    in target_ground_cloud (an open3d point cloud), and use these correspondences to compute the
    similarity transformation using the Umeyama algorithm.
    """
    target_points = np.asarray(target_ground_cloud.points)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_points)
    distances, indices = nbrs.kneighbors(source_ground)
    target_corr = target_points[indices.flatten()]
    T, scale, R, t = umeyama_alignment(source_ground, target_corr)
    return T, scale, R, t

def process_point_cloud(input_file, output_prefix, max_height=2.5, horizontal_radius=4.0, color=[0, 0, 1], adjust_ground=True):
    """
    Processes a single point cloud:
      - Loads the cloud.
      - Detects the ground plane and extracts ground inliers.
      - Aligns the cloud so that the ground becomes level.
      - Computes the ground level from the rotated ground inliers.
      - Segments the vehicle by keeping points with Y in [y_ground - max_height, y_ground].
      - Filters horizontally in the XZ-plane.
      - Paints the final vehicle cloud with the specified color.
      - Writes the final processed cloud to a file.
    Returns the aligned cloud, the final filtered vehicle cloud, the rotated ground cloud, and the computed ground level.
    """
    print(f"\nProcessing {input_file} ...")
    pcd = o3d.io.read_point_cloud(input_file)
    
    # Detect ground plane from original cloud.
    plane_model, inliers = detect_ground_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000)
    ground_cloud = pcd.select_by_index(inliers)
    ground_cloud.paint_uniform_color([1, 0, 0])  # red
    
    # Align the entire cloud.
    R_align = align_point_cloud(pcd, plane_model)
    print("Applied rotation matrix to align ground:")
    print(R_align)
    
    # Also rotate the ground cloud.
    ground_cloud.rotate(R_align, center=(0, 0, 0))
    
    # Compute ground level from rotated ground inliers.
    rotated_ground_points = np.asarray(ground_cloud.points)
    y_ground_aligned = np.median(rotated_ground_points[:, 1])
    print(f"Aligned ground level (median y from ground inliers): {y_ground_aligned:.4f}")
    
    # Segment the vehicle using vertical filtering.
    vehicle_pcd = segment_vehicle_aligned(pcd, y_ground_aligned, max_height=max_height)
    
    # Further filter horizontally.
    vehicle_filtered = filter_horizontal(vehicle_pcd, radius=horizontal_radius)
    
    # Paint the final vehicle cloud.
    vehicle_filtered.paint_uniform_color(color)
    
    return pcd, vehicle_filtered, ground_cloud, y_ground_aligned

def main():
    # Define your two input files.
    file_ref = "./data/ecosport_kiri.ply"   # Reference: ecosport_kiri (fixed)
    file_colmap = "./data/points3D.ply"       # COLMAP points3D (to be adjusted)
    
    # Process the reference cloud.
    aligned_ref, vehicle_ref, ground_ref, ground_level_ref = process_point_cloud(file_ref, "ecosport_kiri",
                                                                                    max_height=2.5,
                                                                                    horizontal_radius=4.0,
                                                                                    color=[0, 1, 1],
                                                                                    adjust_ground=True)
    # Process the COLMAP cloud.
    aligned_colmap, vehicle_colmap, ground_colmap, ground_level_colmap = process_point_cloud(file_colmap, "points3D",
                                                                                              max_height=2.5,
                                                                                              horizontal_radius=4.0,
                                                                                              color=[1, 1, 1],
                                                                                              adjust_ground=True)
    
    # Use the ground level from the reference as the common ground.
    common_ground = ground_level_ref
    print(f"Common ground level (from ecosport_kiri): {common_ground:.4f}")
    
    # Sample ground points from each rotated ground cloud.
    source_ground = sample_ground_points(ground_colmap, num_points=1000)
    target_ground = sample_ground_points(ground_ref, num_points=1000)
    
    # Compute the similarity transformation to map COLMAP ground to reference ground.
    T_sim, scale_sim, R_sim, t_sim = compute_similarity_transformation_from_ground(source_ground, ground_ref)
    print("Computed similarity transformation (from COLMAP to ecosport_kiri):")
    print(T_sim)
    print(f"Scale: {scale_sim:.4f}")
    print("Rotation:")
    print(R_sim)
    print("Translation:")
    print(t_sim)
    
    # Apply the similarity transformation only to the COLMAP aligned cloud (and its vehicle segmentation).
    aligned_colmap.transform(T_sim)
    vehicle_colmap.transform(T_sim)
    
    # Export the final adjusted COLMAP cloud.
    o3d.io.write_point_cloud("points3D_aligned_filtered_common.ply", vehicle_colmap)
    # Export the reference filtered cloud.
    o3d.io.write_point_cloud("ecosport_kiri_aligned_filtered_common.ply", vehicle_ref)
    print("Exported files with common ground: ecosport_kiri_aligned_filtered_common.ply and points3D_aligned_filtered_common.ply")
    
    # Optionally, export the similarity transformation to a text file.
    np.savetxt("similarity_transformation.txt", T_sim, fmt="%.6f")
    print("Saved similarity transformation to similarity_transformation.txt")
    
    # Optionally, visualize them together:
    # o3d.visualization.draw_geometries([aligned_ref, aligned_colmap],
    #                                   window_name="Aligned Clouds with Common Ground",
    #                                   width=800, height=600)

if __name__ == "__main__":
    main()
