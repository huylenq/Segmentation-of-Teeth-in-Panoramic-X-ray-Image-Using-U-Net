"""
Panoramic X-ray Geometry Visualization Module.

Interactive 3D visualizations to help understand the 2D-3D projection
geometry of panoramic radiography and why different regions have
different measurement reliability.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from physical_metrics import (
    DentalRegion,
    PanoMetricPriors,
    X800_DEFAULT_MF,
)


# Region colors based on measurement confidence
REGION_COLORS = {
    DentalRegion.ANTERIOR: 'rgba(255, 80, 80, 0.7)',      # Red (low confidence)
    DentalRegion.CANINE_PREMOLAR: 'rgba(255, 200, 80, 0.7)',  # Orange (medium)
    DentalRegion.MOLAR: 'rgba(80, 200, 80, 0.7)',         # Green (high confidence)
}


def create_xray_geometry_animation(
    title_suffix: str = "",
) -> go.Figure:
    """
    Create a 3D visualization showing how the X-ray source rotates around
    the dental arch and projects onto the detector.

    This explains WHY magnification varies:
    - Source-to-object distance varies by position
    - Beam angle relative to teeth varies by region

    Args:
        title_suffix: Optional suffix for the title (e.g., calibration method name)

    Returns:
        Plotly Figure with animation controls
    """
    # Dental arch parameters
    arch_width = 120  # mm
    arch_depth = 55   # mm

    # Rotation center and radii
    rotation_radius = 100  # mm (distance from center to X-ray source)

    # Generate X-ray source positions at different angles
    angles = np.linspace(-np.pi/3, np.pi/3, 50)  # ±60 degrees
    source_x = rotation_radius * np.sin(angles)
    source_y = -rotation_radius * np.cos(angles) + arch_depth/2
    source_z = np.zeros_like(angles)

    # Detector positions (opposite to source, rotating with it)
    detector_radius = 80  # mm from center
    detector_x = -detector_radius * np.sin(angles)
    detector_y = detector_radius * np.cos(angles) + arch_depth/2
    detector_z = np.zeros_like(angles)

    # Create dental arch
    t = np.linspace(-1, 1, 100)
    arch_x = arch_width/2 * t
    arch_y = arch_depth * (1 - t**2)

    # Sample tooth positions
    tooth_t = np.array([-0.7, -0.4, 0, 0.4, 0.7])
    tooth_x = arch_width/2 * tooth_t
    tooth_y = arch_depth * (1 - tooth_t**2)
    tooth_labels = ['Molar R', 'Premolar R', 'Anterior', 'Premolar L', 'Molar L']

    fig = go.Figure()

    # Add dental arch (static)
    fig.add_trace(
        go.Scatter3d(
            x=arch_x, y=arch_y, z=np.zeros_like(arch_x),
            mode='lines',
            line=dict(color='gray', width=6),
            name='Dental Arch',
        )
    )

    # Add teeth markers (static)
    fig.add_trace(
        go.Scatter3d(
            x=tooth_x, y=tooth_y, z=np.zeros_like(tooth_x),
            mode='markers+text',
            marker=dict(size=10, color=['green', 'orange', 'red', 'orange', 'green']),
            text=tooth_labels,
            textposition='top center',
            name='Teeth',
        )
    )

    # Add patient head outline
    theta_head = np.linspace(0, 2*np.pi, 50)
    head_x = 70 * np.cos(theta_head)
    head_y = 80 * np.sin(theta_head) + arch_depth/2
    fig.add_trace(
        go.Scatter3d(
            x=head_x, y=head_y, z=np.zeros_like(head_x),
            mode='lines',
            line=dict(color='lightgray', width=2, dash='dot'),
            name='Patient Head (approx)',
        )
    )

    # Add source trajectory
    fig.add_trace(
        go.Scatter3d(
            x=source_x, y=source_y, z=source_z,
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.3)', width=2, dash='dash'),
            name='X-ray Source Path',
        )
    )

    # Add detector trajectory
    fig.add_trace(
        go.Scatter3d(
            x=detector_x, y=detector_y, z=detector_z,
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 0.3)', width=2, dash='dash'),
            name='Detector Path',
        )
    )

    # Initial positions (middle)
    mid_idx = len(angles) // 2

    fig.add_trace(
        go.Scatter3d(
            x=[source_x[mid_idx]], y=[source_y[mid_idx]], z=[source_z[mid_idx]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='diamond'),
            name='X-ray Source',
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[detector_x[mid_idx]], y=[detector_y[mid_idx]], z=[detector_z[mid_idx]],
            mode='markers',
            marker=dict(size=15, color='blue', symbol='square'),
            name='Detector',
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[source_x[mid_idx], 0, detector_x[mid_idx]],
            y=[source_y[mid_idx], arch_depth, detector_y[mid_idx]],
            z=[0, 0, 0],
            mode='lines',
            line=dict(color='yellow', width=3),
            name='X-ray Beam',
        )
    )

    # Create animation frames
    frames = []
    for i in range(len(angles)):
        beam_angle = np.arctan2(source_x[i], -source_y[i] + arch_depth/2)
        focal_t = np.sin(beam_angle)
        focal_x = arch_width/2 * focal_t * 0.8
        focal_y = arch_depth * (1 - focal_t**2 * 0.64)

        frame = go.Frame(
            data=[
                go.Scatter3d(x=arch_x, y=arch_y, z=np.zeros_like(arch_x)),
                go.Scatter3d(x=tooth_x, y=tooth_y, z=np.zeros_like(tooth_x)),
                go.Scatter3d(x=head_x, y=head_y, z=np.zeros_like(head_x)),
                go.Scatter3d(x=source_x, y=source_y, z=source_z),
                go.Scatter3d(x=detector_x, y=detector_y, z=detector_z),
                go.Scatter3d(x=[source_x[i]], y=[source_y[i]], z=[source_z[i]]),
                go.Scatter3d(x=[detector_x[i]], y=[detector_y[i]], z=[detector_z[i]]),
                go.Scatter3d(
                    x=[source_x[i], focal_x, detector_x[i]],
                    y=[source_y[i], focal_y, detector_y[i]],
                    z=[0, 0, 0],
                ),
            ],
            name=str(i)
        )
        frames.append(frame)

    fig.frames = frames

    title_text = 'Panoramic X-ray: Rotating Source Geometry'
    if title_suffix:
        title_text += f'<br><sub>{title_suffix}</sub>'
    else:
        title_text += '<br><sub>Click Play to see how source rotates around the dental arch</sub>'

    fig.update_layout(
        height=700,
        width=900,
        title=dict(text=title_text, font=dict(size=14)),
        scene=dict(
            xaxis_title='Left <- X (mm) -> Right',
            yaxis_title='Posterior -> Anterior (mm)',
            zaxis_title='Z (mm)',
            camera=dict(eye=dict(x=0, y=0, z=2.5)),
            aspectmode='data',
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0, x=0.1,
                xanchor='right', yanchor='top',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=100, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0)
                        )]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=mid_idx,
                yanchor='top', xanchor='left',
                currentvalue=dict(prefix='Source Position: ', visible=True, xanchor='right'),
                transition=dict(duration=0),
                pad=dict(b=10, t=50),
                len=0.9, x=0.1, y=0,
                steps=[
                    dict(
                        args=[[str(i)], dict(
                            frame=dict(duration=0, redraw=True),
                            mode='immediate',
                            transition=dict(duration=0)
                        )],
                        label=f'{np.degrees(angles[i]):.0f}deg',
                        method='animate'
                    )
                    for i in range(0, len(angles), 5)
                ]
            )
        ]
    )

    return fig


def create_magnification_chart(
    priors: PanoMetricPriors,
    title_suffix: str = "",
) -> go.Figure:
    """
    Create a bar chart showing magnification factors and uncertainty by region.

    Args:
        priors: Calibration priors containing magnification factors
        title_suffix: Optional suffix for the title (e.g., calibration method name)

    Returns:
        Plotly Figure with bar charts
    """
    mf = priors.magnification_factors

    regions = ['Anterior\n(Incisors)', 'Canine/Premolar', 'Molar']
    dental_regions = [DentalRegion.ANTERIOR, DentalRegion.CANINE_PREMOLAR, DentalRegion.MOLAR]

    h_mfs = []
    h_uncs = []
    for region in dental_regions:
        h_mf, h_unc = mf.get_horizontal_mf(region)
        h_mfs.append(h_mf)
        h_uncs.append(h_unc * 100)

    v_mf = mf.vertical
    v_unc = mf.vertical_uncertainty * 100

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Magnification Factor by Region', 'Measurement Uncertainty by Region'),
        horizontal_spacing=0.15
    )

    colors = ['rgba(255, 80, 80, 0.8)', 'rgba(255, 200, 80, 0.8)', 'rgba(80, 200, 80, 0.8)']

    # Magnification factors
    fig.add_trace(
        go.Bar(
            name='Horizontal MF',
            x=regions, y=h_mfs,
            marker_color=colors,
            text=[f'{mf:.2f}' for mf in h_mfs],
            textposition='outside',
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            name='Vertical MF',
            x=regions, y=[v_mf] * 3,
            marker_color='rgba(100, 100, 200, 0.8)',
            text=[f'{v_mf:.2f}'] * 3,
            textposition='outside',
        ),
        row=1, col=1
    )

    # Uncertainty percentages
    fig.add_trace(
        go.Bar(
            name='Horizontal Uncertainty',
            x=regions, y=h_uncs,
            marker_color=colors,
            text=[f'+/-{u:.0f}%' for u in h_uncs],
            textposition='outside',
            showlegend=False,
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            name='Vertical Uncertainty',
            x=regions, y=[v_unc] * 3,
            marker_color='rgba(100, 100, 200, 0.8)',
            text=[f'+/-{v_unc:.0f}%'] * 3,
            textposition='outside',
            showlegend=False,
        ),
        row=1, col=2
    )

    title_text = 'Panoramic X-ray: Regional Magnification & Uncertainty'
    if title_suffix:
        title_text += f'<br><sub>{title_suffix}</sub>'

    fig.update_layout(
        height=400,
        width=1000,
        title=title_text,
        barmode='group',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.45),
    )

    fig.update_yaxes(title_text='Magnification Factor', range=[1.0, 1.4], row=1, col=1)
    fig.update_yaxes(title_text='Uncertainty (%)', range=[0, 20], row=1, col=2)

    return fig


def create_panoramic_geometry_visualization(
    priors: PanoMetricPriors,
    detected_teeth=None,
    edentulous_zones=None,
    image_width_px: int = 3126,
    image_height_px: int = 1300,
    title_suffix: str = "",
) -> go.Figure:
    """
    Create an interactive 3D visualization of panoramic X-ray projection geometry.

    Shows:
    1. Dental arch as 3D curve with FDI tooth positions
    2. Focal trough concept
    3. Regional magnification zones
    4. How 3D positions map to 2D panoramic image

    Args:
        priors: Calibration priors for magnification factors
        detected_teeth: Optional list of detected teeth from analysis
        edentulous_zones: Optional list of edentulous zones from analysis
        image_width_px: Width of the panoramic image in pixels
        image_height_px: Height of the panoramic image in pixels
        title_suffix: Optional suffix for the title (e.g., calibration method name)

    Returns:
        Plotly Figure with 3D and 2D subplots
    """
    mf = priors.magnification_factors

    # Dental arch geometry
    arch_width = 120  # mm
    arch_depth = 55   # mm

    t = np.linspace(-1, 1, 100)
    arch_x = arch_width/2 * t
    arch_y = arch_depth * (1 - t**2)

    # Tooth positions
    tooth_positions = {
        1: 0.08, 2: 0.18, 3: 0.30, 4: 0.42,
        5: 0.54, 6: 0.68, 7: 0.82, 8: 0.95,
    }

    teeth_3d = []
    for fdi_pos, t_frac in tooth_positions.items():
        for quadrant, (x_sign, z_val) in enumerate([
            (-1, 10),   # Q1: Upper Right
            (1, 10),    # Q2: Upper Left
            (1, -10),   # Q3: Lower Left
            (-1, -10),  # Q4: Lower Right
        ], start=1):
            fdi = quadrant * 10 + fdi_pos
            x = x_sign * arch_width/2 * t_frac
            y = arch_depth * (1 - t_frac**2)

            if fdi_pos <= 2:
                region = DentalRegion.ANTERIOR
            elif fdi_pos <= 5:
                region = DentalRegion.CANINE_PREMOLAR
            else:
                region = DentalRegion.MOLAR

            teeth_3d.append((fdi, x, y, z_val, region))

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.55, 0.45],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=(
            "3D Dental Arch with Regional Magnification",
            "2D Panoramic Projection (Simulated)"
        )
    )

    # 3D: Upper arch
    fig.add_trace(
        go.Scatter3d(
            x=arch_x, y=arch_y, z=np.full_like(arch_x, 10),
            mode='lines',
            line=dict(color='gray', width=4),
            name='Upper Arch',
        ),
        row=1, col=1
    )

    # 3D: Lower arch
    fig.add_trace(
        go.Scatter3d(
            x=arch_x, y=arch_y, z=np.full_like(arch_x, -10),
            mode='lines',
            line=dict(color='gray', width=4, dash='dash'),
            name='Lower Arch',
        ),
        row=1, col=1
    )

    # 3D: Teeth by region
    for region in [DentalRegion.ANTERIOR, DentalRegion.CANINE_PREMOLAR, DentalRegion.MOLAR]:
        region_teeth = [(fdi, x, y, z) for fdi, x, y, z, r in teeth_3d if r == region]
        if region_teeth:
            fdi_nums, xs, ys, zs = zip(*region_teeth)
            h_mf, h_unc = mf.get_horizontal_mf(region)

            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='markers+text',
                    marker=dict(size=8, color=REGION_COLORS[region], line=dict(color='black', width=1)),
                    text=[str(fdi) for fdi in fdi_nums],
                    textposition='top center',
                    textfont=dict(size=8),
                    name=f'{region.value.title()} (MF: {h_mf:.2f} +/-{h_unc*100:.0f}%)',
                    hovertemplate='FDI %{text}<br>X: %{x:.1f}mm<br>Y: %{y:.1f}mm<extra></extra>',
                ),
                row=1, col=1
            )

    # 3D: Focal trough
    theta_trough = np.linspace(0, 2*np.pi, 50)
    trough_inner_x = 0.7 * arch_width/2 * np.cos(theta_trough)
    trough_inner_y = 0.7 * arch_depth * (1 + np.sin(theta_trough))/2
    trough_outer_x = 1.1 * arch_width/2 * np.cos(theta_trough)
    trough_outer_y = 1.1 * arch_depth * (1 + np.sin(theta_trough))/2

    mask = theta_trough < np.pi
    fig.add_trace(
        go.Scatter3d(
            x=np.concatenate([trough_inner_x[mask], trough_outer_x[mask][::-1]]),
            y=np.concatenate([trough_inner_y[mask], trough_outer_y[mask][::-1]]),
            z=np.zeros(sum(mask)*2),
            mode='lines',
            line=dict(color='blue', width=2, dash='dot'),
            name='Focal Trough (midplane)',
        ),
        row=1, col=1
    )

    # 2D projection
    teeth_2d = []
    for fdi, x, y, z, region in teeth_3d:
        pano_x = (x / (arch_width/2) + 1) / 2 * image_width_px
        pano_y = (1 - (z + 10) / 20) * image_height_px
        teeth_2d.append((fdi, pano_x, pano_y, region))

    for region in [DentalRegion.ANTERIOR, DentalRegion.CANINE_PREMOLAR, DentalRegion.MOLAR]:
        region_teeth = [(fdi, px, py) for fdi, px, py, r in teeth_2d if r == region]
        if region_teeth:
            fdi_nums, pxs, pys = zip(*region_teeth)
            h_mf, h_unc = mf.get_horizontal_mf(region)

            fig.add_trace(
                go.Scatter(
                    x=pxs, y=pys,
                    mode='markers+text',
                    marker=dict(size=12, color=REGION_COLORS[region], line=dict(color='black', width=1)),
                    text=[str(fdi) for fdi in fdi_nums],
                    textposition='top center',
                    textfont=dict(size=8),
                    name=f'{region.value.title()} (+/-{h_unc*100:.0f}% horiz)',
                    hovertemplate='FDI %{text}<br>X: %{x:.0f}px<br>Y: %{y:.0f}px<extra></extra>',
                    showlegend=False,
                ),
                row=1, col=2
            )

    # Overlay detected teeth
    if detected_teeth:
        det_x = [t.centroid[0] for t in detected_teeth]
        det_y = [t.centroid[1] for t in detected_teeth]
        det_fdi = [t.fdi_number for t in detected_teeth]

        fig.add_trace(
            go.Scatter(
                x=det_x, y=det_y,
                mode='markers',
                marker=dict(size=8, color='rgba(0, 0, 255, 0.5)', symbol='x', line=dict(width=2)),
                name='Detected Teeth',
                hovertemplate='FDI %{text}<br>X: %{x:.0f}px<br>Y: %{y:.0f}px<extra></extra>',
                text=[str(fdi) for fdi in det_fdi],
            ),
            row=1, col=2
        )

    # Overlay edentulous zones
    if edentulous_zones:
        for zone in edentulous_zones:
            x0, y0, w, h = zone.bbox
            fig.add_shape(
                type='rect',
                x0=x0, y0=y0, x1=x0+w, y1=y0+h,
                line=dict(color='red', width=2, dash='dash'),
                fillcolor='rgba(255, 0, 0, 0.1)',
                row=1, col=2
            )

    title_text = "Panoramic X-ray: 3D->2D Projection Geometry"
    if title_suffix:
        title_text += f"<br><sub>{title_suffix}</sub>"

    fig.update_layout(
        height=600,
        width=1200,
        title=dict(text=title_text, font=dict(size=16)),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.8)'),
        scene=dict(
            xaxis_title='Left <- X (mm) -> Right',
            yaxis_title='Y: Posterior -> Anterior (mm)',
            zaxis_title='Z: Inferior -> Superior (mm)',
            camera=dict(eye=dict(x=0, y=-2, z=0.8)),
            aspectmode='data',
        ),
    )

    fig.update_xaxes(title_text='Panoramic X (pixels)', range=[0, image_width_px], row=1, col=2)
    fig.update_yaxes(title_text='Panoramic Y (pixels)', range=[image_height_px, 0], row=1, col=2)

    return fig


def print_geometry_insights():
    """Print key insights about panoramic X-ray geometry."""
    print("""
KEY INSIGHTS - Panoramic X-ray Geometry:
================================================================================

1. HORIZONTAL measurements (width) are LESS reliable than VERTICAL (height)
   - Vertical uncertainty: +/-3% (consistent across all regions)
   - Horizontal uncertainty: +/-6% (molar) to +/-15% (anterior)

2. ANTERIOR region (incisors) has the HIGHEST horizontal uncertainty
   - X-ray beam is nearly parallel to the dental arch at the midline
   - This causes maximum horizontal distortion

3. MOLAR region has the LOWEST horizontal uncertainty
   - X-ray beam is more perpendicular to teeth in this region
   - Most reliable for mesiodistal width measurements

4. For IMPLANT PLANNING in the anterior region:
   - ALWAYS verify with CBCT
   - Panoramic width measurements can be off by 10-20%

================================================================================
""")
