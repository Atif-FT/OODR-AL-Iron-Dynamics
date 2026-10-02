"""
========================================================================================
Automated OVITO Pro Atomistic Visualization & Publication Figure Pipeline
npj Computational Materials - Revised Manuscript Package
========================================================================================
This script generates genuine 3D atomistic visualizations directly from archived 
molecular dynamics trajectory files:
  1. dynamic_shear.xyz -> Fig_OVITO_Shear.png (Figure 6)
  2. dynamic_shock.xyz -> Fig_OVITO_Shock.png (Figure 8)

Methodology & Standards:
- Genuine OVITO Pro Tachyon ray-tracer with ambient occlusion (24 samples) and soft shadows
- Explicit simulation bounding box wireframe showing physical lattice deformation
- 3D crystallographic orientation tripod (x, y, z)
- Color coding by Atomic Displacement Magnitude (Delta r in Angstroms, Viridis colormap)
- Clean, transparent scientific presentation (no artificial borders, banners, or badges)
- Adheres strictly to Nature Portfolio / npj Computational Materials Q1 publication standards

Usage:
  python render_ovito_cna.py
  (Automatically detects OVITO Pro / ovitos and system Python to render and stitch)
========================================================================================
"""

import os
import sys
import subprocess
import shutil

def render_raw_frames(traj_dir, raw_dir):
    """Render individual frame snapshots using the OVITO Pro Python API."""
    from ovito.io import import_file
    from ovito.modifiers import CalculateDisplacementsModifier, ColorCodingModifier
    from ovito.vis import Viewport, TachyonRenderer, CoordinateTripodOverlay, ColorLegendOverlay, TextLabelOverlay

    os.makedirs(raw_dir, exist_ok=True)

    simulations = [
        {
            "name": "dynamic_shear",
            "file": "dynamic_shear.xyz",
            "max_disp": 1.20,
            "frames": [
                (0, "(a)  t = 0.0 ps (\u03b3 = 0.00)", True),
                (25, "(b)  t = 2.5 ps (\u03b3 = 0.025)", True),
                (50, "(c)  t = 5.0 ps (\u03b3 = 0.050)", True)
            ]
        },
        {
            "name": "dynamic_shock",
            "file": "dynamic_shock.xyz",
            "max_disp": 1.60,
            "frames": [
                (0, "(a)  t = 0.0 ps", True),
                (25, "(b)  t = 2.5 ps", True),
                (50, "(c)  t = 5.0 ps", True)
            ]
        }
    ]

    for sim in simulations:
        xyz_file = os.path.join(traj_dir, sim["file"])
        if not os.path.exists(xyz_file):
            print(f"[ERROR] Trajectory file not found: {xyz_file}")
            continue

        print(f"[OVITO] Loading trajectory: {xyz_file}")
        pipeline = import_file(xyz_file)

        # Displacement calculation relative to reference frame 0
        disp = CalculateDisplacementsModifier()
        pipeline.modifiers.append(disp)

        # Color coding by displacement magnitude using Viridis gradient
        color_mod = ColorCodingModifier(property='Displacement Magnitude')
        color_mod.gradient = ColorCodingModifier.Viridis()
        color_mod.start_value = 0.0
        color_mod.end_value = sim["max_disp"]
        pipeline.modifiers.append(color_mod)
        pipeline.add_to_scene()

        # Visual settings for particles and simulation cell
        data0 = pipeline.compute(0)
        data0.particles.vis.radius = 0.58
        data0.cell.vis.line_width = 0.08
        data0.cell.vis.rendering_color = (0.1, 0.1, 0.1)

        vp = Viewport()
        vp.type = Viewport.Type.Perspective
        vp.camera_pos = (21.0, -18.0, 15.0)
        vp.camera_dir = (-21.0 + 4.25, 18.0 + 4.25, -15.0 + 4.25)
        vp.fov = 0.6
        vp.zoom_all()

        # Coordinate tripod overlay showing x, y, z axes
        tripod = CoordinateTripodOverlay(size=0.08, axis1_label='x', axis2_label='y', axis3_label='z')
        vp.overlays.append(tripod)

        tachyon = TachyonRenderer(
            ambient_occlusion=True,
            ambient_occlusion_brightness=0.92,
            ambient_occlusion_samples=24,
            shadows=True
        )

        for frame_idx, label_text, add_cbar in sim["frames"]:
            # Panel label overlay: (a), (b), (c)
            label = TextLabelOverlay(
                text=label_text,
                alignment=33,  # Top-Left
                font_size=0.045,
                text_color=(0.1, 0.1, 0.1),
                offset_x=0.03,
                offset_y=0.03
            )
            vp.overlays.append(label)

            cbar = None
            if add_cbar:
                cbar = ColorLegendOverlay(
                    modifier=color_mod,
                    title="Atomic Displacement (\u00c5)",
                    format_string="%.2f",
                    alignment=68,  # Bottom-Center
                    font_size=0.040,
                    label_size=0.9,
                    legend_size=0.42,
                    offset_y=0.04
                )
                vp.overlays.append(cbar)

            out_img = os.path.join(raw_dir, f"{sim['name']}_frame_{frame_idx}.png")
            vp.render_image(
                filename=out_img,
                size=(1200, 1000),
                background=(1.0, 1.0, 1.0),
                renderer=tachyon,
                frame=frame_idx
            )
            print(f"[OVITO] Rendered snapshot: {out_img}")

            vp.overlays.remove(label)
            if cbar:
                vp.overlays.remove(cbar)

        pipeline.remove_from_scene()

def compose_figures(raw_dir, output_dirs):
    """Assemble 3-panel publication figures with pure white gutter and copy to destinations."""
    from PIL import Image

    for d in output_dirs:
        os.makedirs(d, exist_ok=True)

    tasks = [
        {
            "name": "dynamic_shear",
            "out_name": "Fig_OVITO_Shear.png",
            "frames": [0, 25, 50]
        },
        {
            "name": "dynamic_shock",
            "out_name": "Fig_OVITO_Shock.png",
            "frames": [0, 25, 50]
        }
    ]

    gutter = 20

    for task in tasks:
        imgs = [Image.open(os.path.join(raw_dir, f"{task['name']}_frame_{f}.png")) for f in task["frames"]]
        w, h = imgs[0].size
        total_w = 3 * w + 2 * gutter
        composite = Image.new("RGB", (total_w, h), (255, 255, 255))

        for i, img in enumerate(imgs):
            composite.paste(img, (i * (w + gutter), 0))

        # Save to raw_dir first
        local_target = os.path.join(raw_dir, task["out_name"])
        composite.save(local_target, dpi=(300, 300))
        print(f"[COMPOSITE] Saved composite image: {local_target}")

        # Copy to all destination directories
        for d in output_dirs:
            dest = os.path.join(d, task["out_name"])
            shutil.copyfile(local_target, dest)
            print(f"[DEPLOY] Copied to: {dest}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    traj_path = os.path.abspath(os.path.join(base_dir, "..", "06_Repositori_Data_Zenodo_Final", "trajectories"))
    raw_frames_dir = os.path.join(script_dir, "raw_cna_renders")

    target_output_dirs = [
        os.path.abspath(os.path.join(base_dir, "..", "01_Naskah_Artikel_Final")),
        os.path.abspath(os.path.join(base_dir, "..", "02_Naskah_Marked_Up_Final")),
        os.path.abspath(os.path.join(base_dir, "..", "04_Surat_Tanggapan_Reviewer_dan_Editor", "figures")),
        os.path.abspath(os.path.join(traj_path, "review_images_cna"))
    ]

    # CLI sub-commands
    if len(sys.argv) > 1 and sys.argv[1] == '--render-only':
        render_raw_frames(traj_path, raw_frames_dir)
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] == '--composite-only':
        compose_figures(raw_frames_dir, target_output_dirs)
        sys.exit(0)

    # Master pipeline:
    # 1. Render via ovitos if ovito not in current environment
    try:
        import ovito
        print("[INFO] OVITO module detected in current interpreter.")
        render_raw_frames(traj_path, raw_frames_dir)
    except ImportError:
        ovitos_exe = shutil.which("ovitos")
        if not ovitos_exe and os.path.exists(r"C:\Program Files\OVITO Pro\ovitos.exe"):
            ovitos_exe = r"C:\Program Files\OVITO Pro\ovitos.exe"

        if ovitos_exe:
            print(f"[INFO] Invoking OVITO Pro interpreter: {ovitos_exe}")
            cmd = [ovitos_exe, os.path.abspath(__file__), "--render-only"]
            subprocess.run(cmd, check=True)
        else:
            print("[ERROR] Neither 'ovito' package nor 'ovitos' executable was found.")
            sys.exit(1)

    # 2. Compose using PIL
    try:
        compose_figures(raw_frames_dir, target_output_dirs)
        print("[SUCCESS] All genuine OVITO publication figures rendered and deployed successfully!")
    except ImportError:
        python_exe = sys.executable
        print(f"[INFO] Invoking Python interpreter for image composition: {python_exe}")
        cmd = [python_exe, os.path.abspath(__file__), "--composite-only"]
        subprocess.run(cmd, check=True)
