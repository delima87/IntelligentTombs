import open3d as o3d
import numpy as np
from utils import *

if __name__ == "__main__":
    # Replace with your actual point cloud file path
    pcd_path = "Scan.ply"  # or .ply, .xyz, etc.
    picked_points ="picking_list.txt"  # Text file containing picked points (x, y, z) for Z section creation
    
    pcd = read_and_downsample(pcd_path, voxel_size=0.05)
    ref_pts = read_picked_points(picked_points)

    # Transform point cloud and picked points with the same rotation
    if ref_pts is not None:
        pcd, grid, ref_pts_transformed = align_main_plane_with_grid(pcd, picked_points=ref_pts)
    else:
        pcd, grid = align_main_plane_with_grid(pcd)
        ref_pts_transformed = None

    if pcd is not None:
        # Create spheres for each picked point
        geometries_to_draw = [pcd, grid]
        
        # Color palette for z sections
        color_palette = [
            [1.0, 0.0, 0.0],      # Red
            [0.0, 1.0, 0.0],      # Green
            [0.0, 0.0, 1.0],      # Blue
            [1.0, 1.0, 0.0],      # Yellow
            [1.0, 0.0, 1.0],      # Magenta
            [0.0, 1.0, 1.0],      # Cyan
            [1.0, 0.5, 0.0],      # Orange
            [0.5, 0.0, 1.0],      # Purple
            [1.0, 0.0, 0.5],      # Pink
            [0.0, 0.5, 1.0],      # Sky Blue
        ]
        
        if ref_pts_transformed is not None:
            # Create z sections for each picked point
            z_sections = []
            for idx, point in enumerate(ref_pts_transformed):
                # Create sphere at picked point
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                sphere.translate(point)
                sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Green color
                geometries_to_draw.append(sphere)
                
                # Create z section at this point's z value
                z_value = point[2]
                color = color_palette[idx % len(color_palette)]
                z_section = create_z_section(pcd, z_value=z_value, z_tolerance=0.05, color=color)
                keypoints = compute_iss_keypoints(z_section,gamma_21=0.3, gamma_32=0.3, min_neighbors=1, color=color)
                # mesh, planes = delaunay_triangulation_mesh(keypoints, return_plane_equations=True)
                lines = connect_keypoints_with_lines(keypoints, max_distance=0.99, k_neighbors=2)
                file_name = "tombsection_" + str(idx + 1) + ".dxf"
                export_to_dxf(file_name, keypoints=keypoints, lines=lines)

                if z_section is not None:
                    z_sections.append(keypoints)
                    geometries_to_draw.append(keypoints)
                    geometries_to_draw.append(lines)
                    print(f"Z section {idx + 1}: Z = {z_value:.4f}, Color = {color}")
            
            print(f"\nAdded {len(ref_pts_transformed)} green spheres for picked points")
            print(f"Created {len(z_sections)} z sections")
        
        o3d.visualization.draw_geometries(geometries_to_draw, window_name="Aligned Point Cloud with Grid and Z Sections")
        
    if ref_pts_transformed is not None:
        print(f"\nTransformed picked points:\n{ref_pts_transformed}")
    # Create a Z section based on user-picked point
    # z_section = create_z_section(pcd, grid, z_tolerance=0.05)
    # if z_section is not None:
    #     o3d.io.write_point_cloud("z_section.ply", z_section)
    #     o3d.io.write_point_cloud("aligned_point_cloud.ply", pcd)
    #     print("Z section saved to z_section.ply")
