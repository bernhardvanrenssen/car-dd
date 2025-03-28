import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors

def load_point_clouds(colmap_path, gsplat_path, voxel_size=0.05):
    """
    Loads two point clouds and downsamples them.
    Returns:
      colmap_down: downsampled COLMAP point cloud (for registration)
      gsplat_down: downsampled gsplat point cloud (for registration)
      colmap_full: original COLMAP point cloud (to be transformed)
      gsplat_full: original gsplat point cloud (for reference)
    """
    colmap_pcd = o3d.io.read_point_cloud(colmap_path)
    gsplat_pcd = o3d.io.read_point_cloud(gsplat_path)
    
    colmap_down = colmap_pcd.voxel_down_sample(voxel_size)
    gsplat_down = gsplat_pcd.voxel_down_sample(voxel_size)
    
    return colmap_down, gsplat_down, colmap_pcd, gsplat_pcd

def compute_correspondences(source_points, target_points, distance_threshold=0.1):
    """
    Computes correspondences between source and target point clouds using nearest neighbors.
    Returns arrays of corresponding source and target points.
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_points)
    distances, indices = nbrs.kneighbors(source_points)
    valid = distances.flatten() < distance_threshold
    source_corr = source_points[valid]
    target_corr = target_points[indices.flatten()][valid]
    return source_corr, target_corr

def umeyama_alignment(source, target):
    """
    Computes the similarity transformation that best aligns the source to the target point cloud.
    Returns a 4x4 transformation matrix T, along with the scale, rotation, and translation.
    """
    # Compute centroids.
    mu_source = np.mean(source, axis=0)
    mu_target = np.mean(target, axis=0)
    
    # Center the point clouds.
    src_centered = source - mu_source
    tgt_centered = target - mu_target
    
    # Compute covariance matrix.
    cov = src_centered.T @ tgt_centered / source.shape[0]
    
    # Perform singular value decomposition.
    U, D, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    
    # Ensure a proper rotation (determinant = +1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    
    # Compute the scale.
    var_src = np.sum(np.linalg.norm(src_centered, axis=1)**2) / source.shape[0]
    scale = np.trace(np.diag(D)) / var_src
    
    # Compute the translation.
    t = mu_target - scale * R @ mu_source
    
    # Construct the homogeneous transformation matrix.
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    
    return T, scale, R, t

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    Executes global registration using RANSAC based on FPFH feature matching.
    Returns the RANSAC registration result.
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        True,  # mutual_filter
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def main():
    # Paths to your point cloud files.
    colmap_path = "./data/vid_trained_colmap_crop.ply"         # COLMAP output PLY (sparse reconstruction)
    gsplat_path = "data/ecosport_kiri.ply"        # gsplat dense PLY file from KIRI engine
    
    # Load and downsample point clouds.
    colmap_down, gsplat_down, colmap_full, gsplat_full = load_point_clouds(colmap_path, gsplat_path, voxel_size=0.05)
    
    # Preprocess: estimate normals and compute FPFH features.
    # For COLMAP:
    colmap_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    gsplat_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        colmap_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        gsplat_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
    
    # Execute global registration using RANSAC.
    result_ransac = execute_global_registration(colmap_down, gsplat_down, source_fpfh, target_fpfh, voxel_size=0.05)
    print("Initial RANSAC Transformation:")
    print(result_ransac.transformation)
    
    # Refine registration with ICP.
    distance_threshold = 0.05 * 0.8  # adjust threshold as needed
    result_icp = o3d.pipelines.registration.registration_icp(
        colmap_full, gsplat_full, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    T_refined = result_icp.transformation
    print("Refined Transformation matrix:")
    print(T_refined)
    
    # Apply the refined transformation to the full COLMAP point cloud.
    colmap_full.transform(T_refined)
    
    # Write the transformed COLMAP point cloud and the gsplat point cloud to new PLY files.
    o3d.io.write_point_cloud("colmap_aligned.ply", colmap_full)
    o3d.io.write_point_cloud("gsplat_dense_out.ply", gsplat_full)
    print("Saved colmap_aligned.ply and gsplat_dense_out.ply")
    
    # Optionally, visualize the two point clouds together (if you have a GUI)
    # o3d.visualization.draw_geometries([colmap_full, gsplat_full])

if __name__ == "__main__":
    main()
