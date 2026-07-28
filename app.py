import os
import traceback

import gradio as gr
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

from utils import (
    align_main_plane_with_grid,
    compute_iss_keypoints,
    connect_keypoints_with_lines,
    create_z_section,
    export_to_dxf,
    read_and_downsample,
    read_picked_points,
)


COLOR_PALETTE = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.5, 0.0],
    [0.5, 0.0, 1.0],
    [1.0, 0.0, 0.5],
    [0.0, 0.5, 1.0],
]

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() in {"1", "true", "yes"}


def rgb_float_to_css(rgb):
    r = int(np.clip(rgb[0], 0.0, 1.0) * 255)
    g = int(np.clip(rgb[1], 0.0, 1.0) * 255)
    b = int(np.clip(rgb[2], 0.0, 1.0) * 255)
    return f"rgb({r},{g},{b})"


def point_cloud_trace(pcd, name, color=None, marker_size=1.5):
    points = np.asarray(pcd.points)
    if points.size == 0:
        return None

    marker = {"size": marker_size, "opacity": 0.85}

    if color is not None:
        marker["color"] = rgb_float_to_css(color)
    elif pcd.has_colors():
        colors = np.asarray(pcd.colors)
        marker["color"] = [rgb_float_to_css(c) for c in colors]
    else:
        marker["color"] = "rgb(185,185,185)"

    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=marker,
        name=name,
    )


def line_set_trace(lineset, name, color=None, width=2):
    points = np.asarray(lineset.points)
    lines = np.asarray(lineset.lines)

    if points.size == 0 or lines.size == 0:
        return None

    x_vals, y_vals, z_vals = [], [], []
    for i0, i1 in lines:
        p0 = points[i0]
        p1 = points[i1]
        x_vals.extend([p0[0], p1[0], None])
        y_vals.extend([p0[1], p1[1], None])
        z_vals.extend([p0[2], p1[2], None])

    return go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode="lines",
        line={
            "color": rgb_float_to_css(color or [1.0, 1.0, 1.0]),
            "width": width,
        },
        name=name,
    )


def build_figure(traces, title):
    fig = go.Figure(data=[t for t in traces if t is not None])
    fig.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
        },
        legend={"x": 0.01, "y": 0.99},
    )
    return fig


def process_scan(scan_file, picking_file):
    if scan_file is None or picking_file is None:
        raise gr.Error("Please upload both Scan .ply and picking_list .txt files.")

    scan_path = scan_file if isinstance(scan_file, str) else scan_file.name
    picking_path = picking_file if isinstance(picking_file, str) else picking_file.name

    if not os.path.exists(scan_path):
        raise gr.Error("Uploaded Scan file could not be found.")
    if not os.path.exists(picking_path):
        raise gr.Error("Uploaded picking list file could not be found.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_lines = []

    try:
        pcd = read_and_downsample(scan_path, voxel_size=0.05)
        ref_pts = read_picked_points(picking_path)

        if ref_pts is not None:
            pcd, grid, ref_pts_transformed = align_main_plane_with_grid(pcd, picked_points=ref_pts)
        else:
            pcd, grid = align_main_plane_with_grid(pcd)
            ref_pts_transformed = None

        scene_traces = []
        section_traces = []

        scene_traces.append(point_cloud_trace(pcd, "Aligned Point Cloud", marker_size=1.2))
        scene_traces.append(line_set_trace(grid, "Grid", color=[0.6, 0.6, 0.6], width=1))

        dxf_files = []
        section_count = 0

        if ref_pts_transformed is not None:
            for idx, point in enumerate(ref_pts_transformed):
                color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
                z_value = float(point[2])

                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                sphere.translate(point)
                sphere_pcd = sphere.sample_points_uniformly(number_of_points=200)
                sphere_pcd.paint_uniform_color([0.0, 1.0, 0.0])
                scene_traces.append(point_cloud_trace(sphere_pcd, f"Picked Point {idx + 1}", marker_size=2.8))

                z_section = create_z_section(pcd, z_value=z_value, z_tolerance=0.05, color=color)
                if z_section is None:
                    log_lines.append(f"Section {idx + 1}: no points found at Z={z_value:.4f}")
                    continue

                keypoints = compute_iss_keypoints(
                    z_section,
                    gamma_21=0.3,
                    gamma_32=0.3,
                    min_neighbors=1,
                    color=color,
                )
                lines = connect_keypoints_with_lines(keypoints, max_distance=0.99, k_neighbors=2, color=color)

                dxf_name = f"tombsection_{idx + 1}.dxf"
                dxf_path = os.path.join(OUTPUT_DIR, dxf_name)
                if export_to_dxf(dxf_path, keypoints=keypoints, lines=lines):
                    dxf_files.append(os.path.abspath(dxf_path))

                keypoint_trace = point_cloud_trace(keypoints, f"Section {idx + 1} Keypoints", marker_size=3.2)
                line_trace = line_set_trace(lines, f"Section {idx + 1} Lines", color=color, width=3)

                scene_traces.append(keypoint_trace)
                scene_traces.append(line_trace)
                section_traces.append(keypoint_trace)
                section_traces.append(line_trace)

                section_count += 1
                log_lines.append(f"Section {idx + 1}: Z={z_value:.4f}, exported {dxf_path}")

        fig_main = build_figure(scene_traces, "Viewer 1: Aligned Point Cloud + Sections")
        fig_sections = build_figure(section_traces, "Viewer 2: Section Keypoints + Lines")

        if section_count == 0:
            log_lines.append("No sections were generated.")
        else:
            log_lines.append(f"Generated {section_count} section(s).")

        if dxf_files:
            log_lines.append(f"DXF exports: {len(dxf_files)} file(s) created.")

        return fig_main, fig_sections, "\n".join(log_lines), "\n".join(dxf_files) if dxf_files else "No DXF files generated."

    except gr.Error:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        raise gr.Error(f"Processing failed: {exc}\n\n{tb}")


with gr.Blocks(title="Tomb Section Processor") as demo:
    gr.Markdown("# Tomb Section Processor")
    gr.Markdown(
        "Upload a point cloud (.ply) and a picking list (.txt), then run processing to preview sections and export DXF files."
    )

    with gr.Row():
        scan_input = gr.File(label="Scan File (.ply)", file_types=[".ply"], type="filepath")
        picking_input = gr.File(label="Picking List (.txt)", file_types=[".txt"], type="filepath")

    run_button = gr.Button("Run Processing", variant="primary")

    with gr.Row():
        viewer_main = gr.Plot(label="Viewer 1")
        viewer_sections = gr.Plot(label="Viewer 2")

    status_box = gr.Textbox(label="Run Log", lines=12)
    dxf_output = gr.Textbox(label="Generated DXF Files", lines=6)

    run_button.click(
        fn=process_scan,
        inputs=[scan_input, picking_input],
        outputs=[viewer_main, viewer_sections, status_box, dxf_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=GRADIO_SHARE,
    )
