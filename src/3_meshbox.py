#!/usr/bin/env python3
"""
Standalone script to create a mesh box covering the car in ecosport_kiri.ply.
The script loads the point cloud, detects and aligns the ground plane,
segments the vehicle point cloud, computes an oriented bounding box, and
exports a TriangleMesh representing the vehicle's bounding box.
"""

import open3d as o3d
import numpy as np


def detect_ground_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    """
    Detect the largest plane (assumed to be the ground) in a point cloud.
    Returns the plane model parameters and inlier indices.
    """
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                               ransac_n=ransac_n,
                                               num_iterations=num_iterations)
    return plane_model, inliers


def compute_rotation_to_align(normal, target_normal=np.array([0, 1, 0])):
    """
    Compute the rotation matrix needed to align a given normal with the target normal.
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
    R_matrix = np.eye(3) + vx + vx @ vx * ((1 - dot_val) / (norm_v ** 2))
    return R_matrix


def align_point_cloud(pcd, plane_model):
    """
    Rotate the point cloud so that the detected ground plane becomes horizontal.
    """
    a, b, c, _ = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    R_matrix = compute_rotation_to_align(normal, target_normal=np.array([0, 1, 0]))
    pcd.rotate(R_matrix, center=(0, 0, 0))
    return R_matrix


def segment_vehicle_aligned(pcd, y_ground, max_height=2.5):
    """
    Segments the vehicle point cloud by selecting points between y_ground and y_ground - max_height.
    """
    points = np.asarray(pcd.points)
    indices = np.where((points[:, 1] >= (y_ground - max_height)) &
                       (points[:, 1] <= y_ground))[0]
    vehicle_pcd = pcd.select_by_index(indices)
    return vehicle_pcd


def filter_horizontal(pcd, radius=4.0):
    """
    Optionally filters the point cloud based on horizontal distance from the median (x, z) point.
    This is useful for isolating the car in case extra points are included.
    """
    points = np.asarray(pcd.points)
    center_x = np.median(points[:, 0])
    center_z = np.median(points[:, 2])
    horizontal_dist = np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 2] - center_z) ** 2)
    indices = np.where(horizontal_dist <= radius)[0]
    filtered_pcd = pcd.select_by_index(indices)
    return filtered_pcd


def create_vehicle_oriented_box_mesh(vehicle_cloud):
    """
    Computes the oriented bounding box of the vehicle and returns a mesh box
    that has the same extent and orientation.
    """
    obb = vehicle_cloud.get_oriented_bounding_box()
    extent = obb.extent  # Width, height, depth.
    # Create a box mesh with the computed extents.
    box_mesh = o3d.geometry.TriangleMesh.create_box(width=extent[0],
                                                    height=extent[1],
                                                    depth=extent[2])
    # The default box spans [0, extent] along each axis.
    # Translate so that its center is at the origin.
    box_mesh.translate(-0.5 * extent)
    # Apply the same rotation as the oriented bounding box.
    box_mesh.rotate(obb.R, center=(0, 0, 0))
    # Finally, translate to the oriented bounding box center.
    box_mesh.translate(obb.center)
    box_mesh.paint_uniform_color([0, 1, 0])  # Green color for visibility.
    return box_mesh


def main():
    # Hardcoded path to the ecosport_kiri.ply point cloud.
    input_file = "./data/ecosport_kiri.ply"

    # Load the point cloud.
    pcd = o3d.io.read_point_cloud(input_file)
    if len(pcd.points) == 0:
        print(f"Error: No points found in {input_file}!")
        return

    # Detect ground plane in the point cloud.
    plane_model, inliers = detect_ground_plane(pcd)
    ground_cloud = pcd.select_by_index(inliers)

    # Align the entire point cloud so that the ground plane is horizontal.
    R_align = align_point_cloud(pcd, plane_model)
    ground_cloud.rotate(R_align, center=(0, 0, 0))  # Apply same rotation to ground cloud.

    # Determine the ground level (median y-coordinate of ground points).
    ground_points = np.asarray(ground_cloud.points)
    y_ground = np.median(ground_points[:, 1])

    # Segment out the vehicle: keep points within a vertical range.
    vehicle_pcd = segment_vehicle_aligned(pcd, y_ground, max_height=2.5)
    vehicle_pcd = filter_horizontal(vehicle_pcd, radius=4.0)

    # Create a mesh (box) that covers the vehicle.
    vehicle_box_mesh = create_vehicle_oriented_box_mesh(vehicle_pcd)

    # Save the mesh to a file.
    output_mesh_file = "ecosport_vehicle_box.ply"
    o3d.io.write_triangle_mesh(output_mesh_file, vehicle_box_mesh)
    print(f"Vehicle bounding box mesh saved to: {output_mesh_file}")


if __name__ == "__main__":
    main()
