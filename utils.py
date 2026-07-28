import open3d as o3d
import numpy as np


def read_and_downsample(filepath, voxel_size=0.05):
    """
    Read a point cloud and downsample it using voxel downsampling.
    """
    pcd = o3d.io.read_point_cloud(filepath)
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Original points: {len(pcd.points)}, Downsampled: {len(pcd_downsampled.points)}")
    return pcd_downsampled


def create_xy_grid(pcd, spacing=0.1, z=None, color=(0.6, 0.6, 0.6)):
    """
    Create a grid of lines parallel to the X and Y axes.

    Args:
        pcd: open3d.geometry.PointCloud
        spacing: distance between grid lines
        z: fixed Z height for the grid (defaults to min Z of the point cloud)
        color: RGB color for the grid lines
    """
    bounds = pcd.get_axis_aligned_bounding_box()
    min_b = bounds.get_min_bound()
    max_b = bounds.get_max_bound()

    if z is None:
        z = float(min_b[2])

    xs = np.arange(min_b[0], max_b[0] + spacing, spacing)
    ys = np.arange(min_b[1], max_b[1] + spacing, spacing)

    points = []
    lines = []

    for y in ys:
        start = [min_b[0], y, z]
        end = [max_b[0], y, z]
        points.extend([start, end])
        lines.append([len(points) - 2, len(points) - 1])

    for x in xs:
        start = [x, min_b[1], z]
        end = [x, max_b[1], z]
        points.extend([start, end])
        lines.append([len(points) - 2, len(points) - 1])

    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    grid.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return grid


def align_main_plane_with_grid(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000, picked_points=None):
        """
        Fit the main plane in the point cloud using RANSAC and transform 
        the point cloud so the plane is parallel to the XY grid.
        
        Optionally transforms picked points using the same transformation.

        Args:
            pcd: PointCloud to process
            distance_threshold: RANSAC distance threshold for plane fitting
            ransac_n: number of points to sample for RANSAC plane fitting
            num_iterations: number of RANSAC iterations
            picked_points: ndarray of shape (N, 3) containing [x, y, z] coordinates (optional)
        
        Returns:
            If picked_points is None:
                tuple: (transformed_pcd, grid_transformed)
            If picked_points is provided:
                tuple: (transformed_pcd, grid_transformed, transformed_picked_points)
        """
        print("Fitting main plane with RANSAC...")
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        a, b, c, d = plane_model
        print(f"Plane model: {a}x + {b}y + {c}z + {d} = 0")
        print(f"Inliers: {len(inliers)}")

        rotation_matrix = compute_plane_to_xy_rotation(plane_model)
        center = pcd.get_center()
        pcd.rotate(rotation_matrix, center=center)

        grid_transformed = create_xy_grid(pcd, spacing=0.1)

        print("Point cloud transformed. Main plane is now parallel to XY grid.")

        # Transform picked points if provided
        if picked_points is not None:
            picked_points = np.array(picked_points)
            # Apply same rotation around the point cloud center
            picked_points_centered = picked_points - center
            picked_points_transformed = picked_points_centered @ rotation_matrix.T
            picked_points_transformed = picked_points_transformed + center
            print(f"Transformed {len(picked_points_transformed)} picked points using the same rotation.")
            return pcd, grid_transformed, picked_points_transformed

        return pcd, grid_transformed


def compute_plane_to_xy_rotation(plane_model):
        """
        Compute rotation matrix to align plane normal with Z-axis.
        This makes the plane parallel to the XY grid.

        Args:
            plane_model: [a, b, c, d] coefficients of plane equation ax + by + cz + d = 0

        Returns:
            3x3 rotation matrix
        """
        a, b, c, d = plane_model
        plane_normal = np.array([a, b, c])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        z_axis = np.array([0, 0, 1])

        if np.allclose(plane_normal, z_axis):
            return np.eye(3)

        if np.allclose(plane_normal, -z_axis):
            return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        axis = np.cross(plane_normal, z_axis)
        axis = axis / np.linalg.norm(axis)

        angle = np.arccos(np.dot(plane_normal, z_axis))

        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        return rotation_matrix


def generate_random_rotation():
        """
        Generate a random rotation matrix using random Euler angles.

        Returns:
            3x3 random rotation matrix
        """
        roll = np.random.uniform(0, 2 * np.pi)
        pitch = np.random.uniform(0, 2 * np.pi)
        yaw = np.random.uniform(0, 2 * np.pi)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx


def create_z_section(pcd, z_value, z_tolerance=0.01, color=None):
        """
        Extract all points at a specific Z level.

        Args:
            pcd: PointCloud to extract section from
            z_value: Specific Z level to extract
            z_tolerance: tolerance for Z level matching (default 0.01)
            color: RGB color for the section (default green [0.0, 1.0, 0.0])

        Returns:
            PointCloud containing points at the selected Z level
        """
        if color is None:
            color = [0.0, 1.0, 0.0]
        
        print(f"Creating Z section at Z = {z_value:.4f}")

        z_section_points = []
        z_section_indices = []

        for i, point in enumerate(np.asarray(pcd.points)):
                if abs(point[2] - z_value) <= z_tolerance:
                        z_section_points.append(point)
                        z_section_indices.append(i)

        if len(z_section_indices) == 0:
            print(f"No points found at Z level {z_value:.4f} with tolerance {z_tolerance}")
            return None

        z_section_cloud = pcd.select_by_index(z_section_indices)
        z_section_cloud.paint_uniform_color(color)

        print(f"Extracted {len(z_section_indices)} points at Z level {z_value:.4f}")

        return z_section_cloud


def compute_iss_keypoints(pcd, salient_radius=None, non_max_radius=None, gamma_21=0.975, gamma_32=0.975, min_neighbors=5, color=None):
        """
        Compute ISS (Intrinsic Shape Signatures) keypoints from a point cloud.

        Args:
            pcd: open3d.geometry.PointCloud - Input point cloud
            salient_radius: float - Radius for salient region detection (default: 4 * voxel_size or auto-computed)
            non_max_radius: float - Radius for non-maximum suppression (default: 3 * voxel_size or auto-computed)
            gamma_21: float - Upper bound on ratio between 2nd and 1st eigenvalue (default: 0.975)
            gamma_32: float - Upper bound on ratio between 3rd and 2nd eigenvalue (default: 0.975)
            min_neighbors: int - Minimum number of neighbors for a keypoint (default: 5)
            color: list/tuple - RGB color for keypoints (default: red [1.0, 0.0, 0.0])

        Returns:
            open3d.geometry.PointCloud - Point cloud containing ISS keypoints
        """
        if color is None:
            color = [1.0, 0.0, 0.0]  # Default red color
        
        if not pcd.has_normals():
            print("Computing normals for ISS keypoint detection...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Auto-compute radii if not provided
        if salient_radius is None or non_max_radius is None:
            # Estimate a reasonable voxel size based on point cloud
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            
            if salient_radius is None:
                salient_radius = 6 * avg_dist
            if non_max_radius is None:
                non_max_radius = 4 * avg_dist
        
        print(f"Computing ISS keypoints with salient_radius={salient_radius:.4f}, non_max_radius={non_max_radius:.4f}")
        
        # Compute ISS keypoints
        keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
            pcd,
            salient_radius=salient_radius,
            non_max_radius=non_max_radius,
            gamma_21=gamma_21,
            gamma_32=gamma_32,
            min_neighbors=min_neighbors
        )
        
        print(f"Detected {len(keypoints.points)} ISS keypoints")
        
        # Color keypoints
        keypoints.paint_uniform_color(color)
        
        return keypoints


def connect_keypoints_with_lines(keypoints, max_distance=None, k_neighbors=None, color=None):
        """
        Connect keypoints with lines based on proximity.

        Args:
            keypoints: open3d.geometry.PointCloud - Point cloud of keypoints
            max_distance: float - Maximum distance to connect keypoints (if None, connects nearest neighbors)
            k_neighbors: int - Number of nearest neighbors to connect to each keypoint (default: 3)
            color: list/tuple - RGB color for lines (default: white [1.0, 1.0, 1.0])

        Returns:
            open3d.geometry.LineSet - Line set connecting keypoints
        """
        if color is None:
            color = [1.0, 1.0, 1.0]  # Default white color
        
        if k_neighbors is None:
            k_neighbors = 3
        
        points = np.asarray(keypoints.points)
        num_points = len(points)
        
        if num_points < 2:
            print("Not enough keypoints to connect (need at least 2)")
            return o3d.geometry.LineSet()
        
        # Build KD tree for neighbor search
        pcd_tree = o3d.geometry.KDTreeFlann(keypoints)
        
        lines = []
        for i in range(num_points):
            # Find k nearest neighbors (including itself, so k+1)
            [k, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k_neighbors + 1)
            
            # Connect to neighbors (skip first index which is the point itself)
            for j in idx[1:]:
                if j > i:  # Avoid duplicate lines
                    if max_distance is None or np.linalg.norm(points[i] - points[j]) <= max_distance:
                        lines.append([i, j])
        
        # Create LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
        
        print(f"Connected {num_points} keypoints with {len(lines)} lines")
        
        return line_set


def delaunay_triangulation_mesh(keypoints, return_plane_equations=False):
        """
        Create a triangular mesh from keypoints using Delaunay triangulation.
        
        Each triangle is a planar face. Optionally returns the plane equation for each triangle.

        Args:
            keypoints: open3d.geometry.PointCloud - Point cloud of keypoints
            return_plane_equations: bool - If True, also return plane equations for each triangle

        Returns:
            If return_plane_equations is False:
                open3d.geometry.TriangleMesh - Mesh created from Delaunay triangulation
            If return_plane_equations is True:
                tuple: (mesh, plane_equations_list)
                    plane_equations_list: list of dicts with 'coefficients' and 'equation'
        """
        points = np.asarray(keypoints.points)
        num_points = len(points)
        
        if num_points < 4:
            print(f"Error: Delaunay triangulation requires at least 4 points, got {num_points}")
            return None
        
        print(f"Creating Delaunay triangulation mesh from {num_points} keypoints...")
        
        # Create Delaunay triangulation from points
        # Note: Open3D's Delaunay is limited, so we use scipy for better results
        try:
            from scipy.spatial import Delaunay
            
            # Project points to 2D (XY plane) for 2.5D Delaunay, or use 3D Delaunay
            # For most tomb/cross-section cases, 2D projection on XY works well
            points_2d = points[:, :2]
            
            tri = Delaunay(points_2d)
            triangles = tri.simplices
            
            print(f"Created {len(triangles)} triangles from Delaunay triangulation")
        
        except ImportError:
            print("scipy not available, using Open3D's Delaunay...")
            # Fallback to Open3D's convex hull (not ideal but works)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(keypoints, depth=9)
            triangles = np.asarray(mesh.triangles)
        
        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # Compute vertex normals for better visualization
        mesh.compute_vertex_normals()
        
        # Color the mesh with a nice color
        mesh.paint_uniform_color([0.5, 0.8, 1.0])  # Light blue
        
        if return_plane_equations:
            plane_equations = []
            
            # Compute plane equation for each triangle
            for tri_idx, triangle in enumerate(triangles):
                p0 = points[triangle[0]]
                p1 = points[triangle[1]]
                p2 = points[triangle[2]]
                
                # Compute normal using cross product
                v1 = p1 - p0
                v2 = p2 - p0
                normal = np.cross(v1, v2)
                
                # Normalize
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    normal = normal / norm
                else:
                    normal = np.array([0, 0, 1])  # Fallback
                
                # Plane equation: ax + by + cz + d = 0
                # d = -(ax0 + by0 + cz0)
                a, b, c = normal
                d = -np.dot(normal, p0)
                
                plane_eq = {
                    'triangle_index': tri_idx,
                    'vertices': [triangle[0], triangle[1], triangle[2]],
                    'coefficients': [a, b, c, d],
                    'equation': f"{a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0",
                    'normal': normal.tolist()
                }
                plane_equations.append(plane_eq)
            
            print(f"Computed plane equations for {len(plane_equations)} triangles")
            return mesh, plane_equations
        
        print("Delaunay triangulation mesh created successfully")
        return mesh


def export_to_dxf(filepath, keypoints=None, lines=None, mesh=None, keypoint_size=0.05):
        """
        Export keypoints, lines, and/or mesh to DXF format for AutoCAD.

        Args:
            filepath: str - Path to save the DXF file
            keypoints: open3d.geometry.PointCloud - Point cloud of keypoints (optional)
            lines: open3d.geometry.LineSet - Line set connecting keypoints (optional)
            mesh: open3d.geometry.TriangleMesh - Triangle mesh (optional)
            keypoint_size: float - Size of keypoint circles (default: 0.05)

        Returns:
            bool - True if export successful, False otherwise
        """
        try:
            import ezdxf
        except ImportError:
            print("Error: ezdxf library not found. Install with: pip install ezdxf")
            return False
        
        try:
            # Create a new DXF document
            doc = ezdxf.new(dxfversion='R2010')
            msp = doc.modelspace()
            
            # Export keypoints as points or small circles
            if keypoints is not None:
                points = np.asarray(keypoints.points)
                colors = np.asarray(keypoints.colors) if keypoints.has_colors() else None
                
                for idx, point in enumerate(points):
                    # Add point entity
                    msp.add_point(point)
                    
                    # Optionally add small circles at keypoint locations for visibility
                    msp.add_circle(center=(point[0], point[1], point[2]), radius=keypoint_size, dxfattribs={'layer': 'keypoints'})
                
                print(f"Exported {len(points)} keypoints")
            
            # Export lines
            if lines is not None:
                line_points = np.asarray(lines.points)
                line_indices = np.asarray(lines.lines)
                
                for line_idx, (start_idx, end_idx) in enumerate(line_indices):
                    p1 = line_points[start_idx]
                    p2 = line_points[end_idx]
                    msp.add_line(tuple(p1), tuple(p2), dxfattribs={'layer': 'lines'})
                
                print(f"Exported {len(line_indices)} lines")
            
            # Export mesh triangles as 3D faces
            if mesh is not None:
                vertices = np.asarray(mesh.vertices)
                triangles = np.asarray(mesh.triangles)
                
                for tri_idx, triangle in enumerate(triangles):
                    p1 = vertices[triangle[0]]
                    p2 = vertices[triangle[1]]
                    p3 = vertices[triangle[2]]
                    
                    # Add 3D face (triangle)
                    points_3d = [tuple(p1), tuple(p2), tuple(p3)]
                    msp.add_lwpolyline(
                        [(p1[0], p1[1], p1[2]), (p2[0], p2[1], p2[2]), (p3[0], p3[1], p3[2]), (p1[0], p1[1], p1[2])],
                        dxfattribs={'layer': 'mesh'}
                    )
                
                print(f"Exported {len(triangles)} mesh triangles")
            
            # Create layers for organization
            if 'keypoints' not in doc.layers:
                doc.layers.new(name='keypoints', dxfattribs={'color': 1})  # Red
            if 'lines' not in doc.layers:
                doc.layers.new(name='lines', dxfattribs={'color': 2})  # Yellow
            if 'mesh' not in doc.layers:
                doc.layers.new(name='mesh', dxfattribs={'color': 5})  # Blue
            
            # Save the DXF file
            doc.saveas(filepath)
            print(f"\nDXF file successfully saved to: {filepath}")
            return True
        
        except Exception as e:
            print(f"Error exporting to DXF: {e}")
            return False


def read_picked_points(filepath):
    """
    Read picked points from a text file.
    
    Supports multiple formats:
    1. Three columns (x y z): x y z
    2. Comma-separated (x, y, z): x, y, z
    3. Whitespace-separated: x  y  z
    
    Args:
        filepath: str - Path to the text file containing picked points
    
    Returns:
        ndarray - Array of shape (N, 3) containing [x, y, z] coordinates
                  Returns None if file doesn't exist or is empty
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"File '{filepath}' is empty.")
            return None
        
        points = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            try:
                # Try comma-separated format first
                if ',' in line:
                    coords = [float(x.strip()) for x in line.split(',')]
                else:
                    # Try whitespace-separated format
                    coords = [float(x) for x in line.split()]
                
                if len(coords) != 3:
                    print(f"Warning: Line {line_num} has {len(coords)} values, expected 3. Skipping: {line}")
                    continue
                
                points.append(coords)
            
            except ValueError as e:
                print(f"Warning: Could not parse line {line_num}: '{line}' ({e})")
                continue
        
        if not points:
            print(f"No valid points found in '{filepath}'.")
            return None
        
        points_array = np.array(points, dtype=np.float64)
        print(f"Successfully read {len(points_array)} points from '{filepath}'")
        return points_array
    
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        return None