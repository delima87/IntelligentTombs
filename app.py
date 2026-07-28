import os
import shutil
import socket
import tempfile
import traceback
import uuid
import zipfile

import gradio as gr
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import plotly.io as pio

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


def _get_available_port(start_port: int = 7860):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((SERVER_NAME, start_port))
            return start_port
        except OSError:
            for port in range(start_port + 1, start_port + 20):
                try:
                    sock.bind((SERVER_NAME, port))
                    return port
                except OSError:
                    continue
            return start_port


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


def build_figure(traces, title, show_legend=True):
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
        showlegend=show_legend,
        legend={"x": 0.01, "y": 0.99},
    )
    return fig


def numpy_points_trace(points, name, color=None, marker_size=3.0):
    arr = np.asarray(points)
    if arr.size == 0:
        return None

    marker = {"size": marker_size, "opacity": 0.9}
    marker["color"] = rgb_float_to_css(color or [0.0, 1.0, 0.0])

    return go.Scatter3d(
        x=arr[:, 0],
        y=arr[:, 1],
        z=arr[:, 2],
        mode="markers",
        marker=marker,
        name=name,
    )


def write_pipeline_steps_html(step_figures, output_path):
    sections = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "  <meta charset=\"utf-8\" />",
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
        "  <title>Pipeline Step Figures</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 0; padding: 24px; background: #f7f7f7; }",
        "    h1 { margin: 0 0 20px; }",
        "    .panel { background: white; border: 1px solid #ddd; border-radius: 10px; padding: 12px; margin-bottom: 18px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Point Cloud Processing Pipeline</h1>",
    ]

    for idx, fig in enumerate(step_figures):
        include_plotly = "cdn" if idx == 0 else False
        fig_html = pio.to_html(fig, include_plotlyjs=include_plotly, full_html=False)
        sections.append("  <div class=\"panel\">")
        sections.append(fig_html)
        sections.append("  </div>")

    sections.extend(["</body>", "</html>"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))


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
        pcd_downsampled = read_and_downsample(scan_path, voxel_size=0.05)
        pcd = o3d.geometry.PointCloud(pcd_downsampled)
        ref_pts = read_picked_points(picking_path)

        if ref_pts is not None:
            pcd, grid, ref_pts_transformed = align_main_plane_with_grid(pcd, picked_points=ref_pts)
        else:
            pcd, grid = align_main_plane_with_grid(pcd)
            ref_pts_transformed = None

        scene_traces = []
        section_traces = []
        step3_traces = []
        step4_traces = []
        step5_traces = []

        scene_traces.append(point_cloud_trace(pcd, "Aligned Point Cloud", marker_size=1.2))
        scene_traces.append(line_set_trace(grid, "Grid", color=[0.6, 0.6, 0.6], width=1))

        step1_fig = build_figure(
            [point_cloud_trace(pcd_downsampled, "Downsampled Point Cloud", marker_size=1.3)],
            "Step 1: read_and_downsample",
            show_legend=False,
        )

        step2_traces = [
            point_cloud_trace(pcd, "Aligned Point Cloud", marker_size=1.2),
            line_set_trace(grid, "Grid", color=[0.6, 0.6, 0.6], width=1),
        ]
        if ref_pts_transformed is not None:
            step2_traces.append(
                numpy_points_trace(ref_pts_transformed, "Picked Points", color=[0.0, 1.0, 0.0], marker_size=4.0)
            )
            step3_traces.append(
                numpy_points_trace(ref_pts_transformed, "Picked Points", color=[0.0, 1.0, 0.0], marker_size=4.0)
            )

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
                    gamma_21=0.5,
                    gamma_32=0.5,
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
                z_section_trace = point_cloud_trace(z_section, f"Section {idx + 1} Z Section", marker_size=2.0)

                scene_traces.append(keypoint_trace)
                scene_traces.append(line_trace)
                section_traces.append(keypoint_trace)
                section_traces.append(line_trace)
                step3_traces.append(z_section_trace)
                step4_traces.append(keypoint_trace)
                step5_traces.append(keypoint_trace)
                step5_traces.append(line_trace)

                section_count += 1
                log_lines.append(f"Section {idx + 1}: Z={z_value:.4f}, exported {dxf_path}")

        fig_main = build_figure(scene_traces, "Viewer 1: Aligned Point Cloud + Sections")
        fig_sections = build_figure(section_traces, "Viewer 2: Section Keypoints + Lines")

        step2_fig = build_figure(step2_traces, "Step 2: align_main_plane_with_grid", show_legend=False)
        step3_fig = build_figure(step3_traces, "Step 3: create_z_section", show_legend=False)
        step4_fig = build_figure(step4_traces, "Step 4: compute_iss_keypoints", show_legend=False)
        step5_fig = build_figure(step5_traces, "Step 5: connect_keypoints_with_lines", show_legend=False)
        pipeline_html_path = os.path.join(OUTPUT_DIR, "pipeline_steps.html")
        write_pipeline_steps_html([step1_fig, step2_fig, step3_fig, step4_fig, step5_fig], pipeline_html_path)

        fig_main.write_html(os.path.join(OUTPUT_DIR, "viewer_main.html"), include_plotlyjs="cdn")
        fig_sections.write_html(os.path.join(OUTPUT_DIR, "viewer_sections.html"), include_plotlyjs="cdn")
        log_lines.append(f"Pipeline figures exported: {pipeline_html_path}")

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


def build_output_archive():
    if not os.path.isdir(OUTPUT_DIR):
        raise gr.Error("No output directory found. Run processing first.")

    files_to_package = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for filename in files:
            files_to_package.append(os.path.join(root, filename))

    if not files_to_package:
        raise gr.Error("No processed files available to download yet.")

    archive_name = f"tomb_outputs_{uuid.uuid4().hex[:8]}.zip"
    archive_path = os.path.join(tempfile.gettempdir(), archive_name)

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files_to_package:
            rel_path = os.path.relpath(file_path, OUTPUT_DIR)
            archive.write(file_path, arcname=os.path.join("outputs", rel_path))

    return archive_path


def clear_for_new_pointcloud():
    if os.path.isdir(OUTPUT_DIR):
        for entry in os.scandir(OUTPUT_DIR):
            if entry.is_file() or entry.is_symlink():
                os.remove(entry.path)
            elif entry.is_dir():
                shutil.rmtree(entry.path)

    return (
        None,
        None,
        None,
        None,
        "Cleared all inputs, previews, and generated files. Ready for a new point cloud.",
        "",
        None,
    )


with gr.Blocks(title="Tomb Section Processor") as demo:
    gr.Markdown("# Tomb Section Processor")
    gr.Markdown(
        "Upload a point cloud (.ply) and a picking list (.txt), then run processing to preview sections and export DXF files."
    )

    with gr.Row():
        scan_input = gr.File(label="Scan File (.ply)", file_types=[".ply"], type="filepath")
        picking_input = gr.File(label="Picking List (.txt)", file_types=[".txt"], type="filepath")

    with gr.Row():
        run_button = gr.Button("Run Processing", variant="primary")
        download_button = gr.Button("Download Output Folder")
        clear_button = gr.Button("Clear All / New Point Cloud")

    with gr.Row():
        viewer_main = gr.Plot(label="Viewer 1")
        viewer_sections = gr.Plot(label="Viewer 2")

    status_box = gr.Textbox(label="Run Log", lines=12)
    dxf_output = gr.Textbox(label="Generated DXF Files", lines=6)
    output_archive = gr.File(label="Processed Output Archive (.zip)")

    run_button.click(
        fn=process_scan,
        inputs=[scan_input, picking_input],
        outputs=[viewer_main, viewer_sections, status_box, dxf_output],
    )

    download_button.click(
        fn=build_output_archive,
        outputs=[output_archive],
    )

    clear_button.click(
        fn=clear_for_new_pointcloud,
        outputs=[scan_input, picking_input, viewer_main, viewer_sections, status_box, dxf_output, output_archive],
    )


if __name__ == "__main__":
    resolved_port = _get_available_port(SERVER_PORT)
    demo.launch(
        server_name=SERVER_NAME,
        server_port=resolved_port,
        share=GRADIO_SHARE,
    )
