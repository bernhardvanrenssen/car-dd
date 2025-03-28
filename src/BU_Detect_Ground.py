import open3d as o3d
import numpy as np

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
    # Rotate 90° about X so that the box's front face (originally +z) becomes +y.
    R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
    plane_mesh.rotate(R, center=(0, 0, 0))
    plane_mesh.translate(np.array([0, y_ground, 0]))
    plane_mesh.paint_uniform_color([0.8, 0.8, 0.8])
    return plane_mesh

def segment_vehicle_aligned(pcd, y_ground, max_height=2.5):
    """
    Assumes the point cloud has been rotated so that the ground is level (Y-up).
    For an inverted system (here we assume after alignment Y increases upward),
    we define the ground level as y_ground and keep points with Y in:
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
      - If adjust_ground is True, returns the computed ground level.
      - Writes the final processed cloud to a file.
    Returns the aligned cloud, the final filtered vehicle cloud, and the computed ground level.
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
    
    # Write out the final processed cloud.
    # output_file = f"{output_prefix}_aligned_filtered.ply"
    # o3d.io.write_point_cloud(output_file, vehicle_filtered)
    # print(f"Saved {output_file}")
    
    if adjust_ground:
        return pcd, vehicle_filtered, y_ground_aligned
    else:
        return pcd, vehicle_filtered, None

def main():
    # Define your two input files.
    file1 = "./data/ecosport_kiri.ply"   # KIRI engine dense model (reference, fixed)
    file2 = "./data/points3D.ply"         # COLMAP points3D (to be adjusted)
    
    # Process ecosport_kiri without ground adjustment (it remains fixed).
    aligned_ref, vehicle_ref, ground_ref = process_point_cloud(file1, "ecosport_kiri", max_height=2.5, horizontal_radius=4.0, color=[0, 0, 1], adjust_ground=True)
    
    # Process points3D.
    aligned_colmap, vehicle_colmap, ground_colmap = process_point_cloud(file2, "points3D", max_height=2.5, horizontal_radius=4.0, color=[1, 0, 0], adjust_ground=True)
    
    # Use the ground level from ecosport_kiri as the common reference.
    common_ground = ground_ref
    print(f"Common ground level (reference from ecosport_kiri): {common_ground:.4f}")
    
    # Adjust only the COLMAP (points3D) cloud to match the common ground.
    def adjust_to_common_ground(pcd, current_ground, common_ground):
        translation = np.array([0, common_ground - current_ground, 0])
        pcd.translate(translation, relative=True)
    
    adjust_to_common_ground(aligned_colmap, ground_colmap, common_ground)
    adjust_to_common_ground(vehicle_colmap, ground_colmap, common_ground)
    
    # Export the final adjusted COLMAP cloud.
    o3d.io.write_point_cloud("points3D_aligned_filtered_common.ply", vehicle_colmap)
    # Also export the ecosport_kiri filtered cloud (which remains unmodified).
    o3d.io.write_point_cloud("ecosport_kiri_aligned_filtered_common.ply", vehicle_ref)
    print("Exported files with common ground: ecosport_kiri_aligned_filtered_common.ply and points3D_aligned_filtered_common.ply")
    
    # Optionally, visualize them together:
    # o3d.visualization.draw_geometries([aligned_ref, aligned_colmap],
    #                                   window_name="Aligned Clouds with Common Ground",
    #                                   width=800, height=600)

if __name__ == "__main__":
    main()
