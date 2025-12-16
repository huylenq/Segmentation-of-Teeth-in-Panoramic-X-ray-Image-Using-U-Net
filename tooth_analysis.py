"""
Tooth Analysis Module for Edentulous Space Measurement with FDI Identification.

This module provides:
- Structured tooth component extraction from segmentation masks
- FDI (Fédération Dentaire Internationale) tooth numbering
- Edentulous zone detection and measurement
- Physical (mm) measurement conversion with uncertainty quantification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import cv2
import numpy as np
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans

if TYPE_CHECKING:
    from physical_metrics import PanoMetricPriors, EdentulousZonePhysical


@dataclass
class ToothComponent:
    """Represents a single detected tooth."""
    label_id: int                          # Connected component label
    centroid: tuple[float, float]          # (cx, cy) in pixels
    bbox: tuple[int, int, int, int]        # (x, y, w, h)
    area: int                              # Pixel area
    contour: np.ndarray                    # OpenCV contour points
    width: float                           # Bounding box width
    height: float                          # Bounding box height
    fdi_number: int | None = None          # Assigned FDI number (11-48)
    quadrant: int | None = None            # Quadrant (1-4)


@dataclass
class EdentulousZone:
    """Represents a gap where teeth are missing."""
    quadrant: int                          # 1-4
    missing_fdi_numbers: list[int]         # e.g., [14, 15]
    adjacent_teeth: tuple[int | None, int | None]  # FDI numbers of neighbors
    bbox: tuple[int, int, int, int]        # (x, y, width, height) in pixels
    area: int                              # pixels^2
    missing_count: int                     # Number of teeth that could fit


@dataclass
class DentalAnalysisResult:
    """Complete analysis result for a dental panoramic X-ray."""
    detected_teeth: list[ToothComponent]
    edentulous_zones: list[EdentulousZone]
    reference_count: int                   # 28 or 32
    total_detected: int
    total_missing: int
    quadrant_info: dict = field(default_factory=dict)  # Per-quadrant stats


def extract_tooth_components(
    prediction_mask: np.ndarray,
    erode_iterations: int = 2,
    open_iterations: int = 3,
    min_area: int = 2000
) -> list[ToothComponent]:
    """
    Extract individual tooth components from a segmentation mask.

    This is a refactored version of CCA_Analysis that returns structured data
    instead of just an annotated image.

    Args:
        prediction_mask: BGR prediction image from the model
        erode_iterations: Erosion iterations for separating touching teeth
        open_iterations: Morphological opening iterations
        min_area: Minimum pixel area to consider as a tooth

    Returns:
        List of ToothComponent objects
    """
    # Morphological preprocessing (same as original CCA)
    kernel = np.ones((5, 5), dtype=np.float32)
    kernel_sharpening = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])

    image = prediction_mask.copy()
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    image = cv2.filter2D(image, -1, kernel_sharpening)
    image = cv2.erode(image, kernel, iterations=erode_iterations)

    # Convert to grayscale and threshold
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Connected component analysis
    num_labels, labels = cv2.connectedComponents(thresh, connectivity=8)

    teeth = []
    for label_id in range(1, num_labels):  # Skip background (0)
        # Create mask for this component
        mask = np.zeros(thresh.shape, dtype=np.uint8)
        mask[labels == label_id] = 255

        # Find contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = contours[0]

        # Calculate area
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate centroid using moments
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = x + w / 2, y + h / 2

        # Use axis-aligned bounding box for width/height
        # This is more reliable than rotated rectangle for gap calculations
        # w = horizontal extent (mesiodistal width)
        # h = vertical extent (crown-root height)
        tooth = ToothComponent(
            label_id=label_id,
            centroid=(cx, cy),
            bbox=(x, y, w, h),
            area=area,
            contour=contour,
            width=w,   # Axis-aligned horizontal extent
            height=h,  # Axis-aligned vertical extent
        )
        teeth.append(tooth)

    return teeth


def _separate_arches_kmeans(teeth: list[ToothComponent], image_height: int) -> tuple[list[ToothComponent], list[ToothComponent]]:
    """
    Separate teeth into upper and lower arches using K-Means clustering.

    Args:
        teeth: List of detected teeth
        image_height: Height of the image for fallback

    Returns:
        (upper_arch_teeth, lower_arch_teeth)
    """
    if len(teeth) < 4:
        # Fallback to 50% midline
        y_mid = image_height / 2
        upper = [t for t in teeth if t.centroid[1] < y_mid]
        lower = [t for t in teeth if t.centroid[1] >= y_mid]
        return upper, lower

    # Extract Y-coordinates
    y_coords = np.array([[t.centroid[1]] for t in teeth])

    # K-Means with k=2
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(y_coords)

    # Determine which cluster is upper (lower Y value = top of image)
    cluster_means = [y_coords[labels == i].mean() for i in range(2)]
    upper_cluster = 0 if cluster_means[0] < cluster_means[1] else 1

    upper = [t for t, lbl in zip(teeth, labels) if lbl == upper_cluster]
    lower = [t for t, lbl in zip(teeth, labels) if lbl != upper_cluster]

    return upper, lower


def _separate_quadrants_kmeans(arch_teeth: list[ToothComponent], image_width: int) -> tuple[list[ToothComponent], list[ToothComponent], float]:
    """
    Separate arch teeth into left and right quadrants using K-Means.

    Args:
        arch_teeth: Teeth from one arch
        image_width: Width of the image for fallback

    Returns:
        (left_teeth, right_teeth, x_midline)
    """
    if len(arch_teeth) < 2:
        x_mid = image_width / 2
        left = [t for t in arch_teeth if t.centroid[0] < x_mid]
        right = [t for t in arch_teeth if t.centroid[0] >= x_mid]
        return left, right, x_mid

    # Extract X-coordinates
    x_coords = np.array([[t.centroid[0]] for t in arch_teeth])

    # K-Means with k=2
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(x_coords)

    # Calculate midline as midpoint between cluster centers
    centers = kmeans.cluster_centers_.flatten()
    x_mid = (centers[0] + centers[1]) / 2

    # Left = lower X (image left = patient's right)
    left_cluster = 0 if centers[0] < centers[1] else 1

    left = [t for t, lbl in zip(arch_teeth, labels) if lbl == left_cluster]
    right = [t for t, lbl in zip(arch_teeth, labels) if lbl != left_cluster]

    return left, right, x_mid


def assign_fdi_numbers(
    teeth: list[ToothComponent],
    image_shape: tuple[int, int],
    gap_threshold_multiplier: float = 1.8
) -> list[ToothComponent]:
    """
    Assign FDI tooth numbers based on position.

    FDI System:
    - Quadrant 1 (upper right): 11-18 (patient's right = image left)
    - Quadrant 2 (upper left): 21-28 (patient's left = image right)
    - Quadrant 3 (lower left): 31-38
    - Quadrant 4 (lower right): 41-48

    Args:
        teeth: List of ToothComponent objects
        image_shape: (height, width) of the image
        gap_threshold_multiplier: Spacing threshold for detecting gaps between teeth.
            If spacing > multiplier * median_spacing, a gap is detected. Default 1.8.

    Returns:
        Same list with fdi_number and quadrant fields populated
    """
    if not teeth:
        return teeth

    height, width = image_shape

    # Step 1: Separate upper/lower arches
    upper_teeth, lower_teeth = _separate_arches_kmeans(teeth, height)

    # Step 2: Separate into quadrants
    # Upper arch
    upper_left_img, upper_right_img, upper_x_mid = _separate_quadrants_kmeans(upper_teeth, width)
    # Image left = patient right (Q1), Image right = patient left (Q2)
    q1_teeth = upper_left_img   # Upper right (patient perspective)
    q2_teeth = upper_right_img  # Upper left (patient perspective)

    # Lower arch
    lower_left_img, lower_right_img, lower_x_mid = _separate_quadrants_kmeans(lower_teeth, width)
    # Image left = patient right (Q4), Image right = patient left (Q3)
    q4_teeth = lower_left_img   # Lower right (patient perspective)
    q3_teeth = lower_right_img  # Lower left (patient perspective)

    # Step 3: Order teeth within each quadrant and assign FDI numbers with gap detection
    def assign_quadrant_with_gaps(quadrant_teeth: list[ToothComponent], quadrant: int, x_mid: float, threshold: float):
        """Assign FDI numbers within a quadrant, accounting for gaps."""
        if not quadrant_teeth:
            return

        # Sort by distance from midline (closest = position 1)
        # Q1/Q4: sort descending X (teeth on image-left side, sort right-to-left toward midline)
        # Q2/Q3: sort ascending X (teeth on image-right side, sort left-to-right toward midline)
        if quadrant in [1, 4]:
            sorted_teeth = sorted(quadrant_teeth, key=lambda t: -t.centroid[0])
        else:
            sorted_teeth = sorted(quadrant_teeth, key=lambda t: t.centroid[0])

        # Calculate average tooth width and spacing in this quadrant
        if len(sorted_teeth) < 2:
            for i, tooth in enumerate(sorted_teeth, start=1):
                tooth.quadrant = quadrant
                tooth.fdi_number = quadrant * 10 + i
            return

        avg_width = np.mean([t.width for t in sorted_teeth])

        # Calculate spacings between adjacent teeth
        spacings = []
        for i in range(len(sorted_teeth) - 1):
            t1, t2 = sorted_teeth[i], sorted_teeth[i + 1]
            spacing = abs(t2.centroid[0] - t1.centroid[0])
            spacings.append(spacing)

        # Median spacing is the "normal" spacing (robust to outliers)
        median_spacing = np.median(spacings) if spacings else avg_width

        # Assign FDI numbers, incrementing position when gaps are detected
        current_pos = 1
        sorted_teeth[0].quadrant = quadrant
        sorted_teeth[0].fdi_number = quadrant * 10 + current_pos

        for i in range(1, len(sorted_teeth)):
            spacing = spacings[i - 1]

            # If spacing is significantly larger than normal, there's a gap
            # Estimate how many teeth could fit in the gap
            if spacing > threshold * median_spacing:
                # Calculate number of missing teeth in this gap
                missing_count = max(1, round((spacing - threshold * median_spacing) / median_spacing))
                current_pos += missing_count

            current_pos += 1

            # Cap at position 8 (max teeth per quadrant)
            if current_pos > 8:
                current_pos = 8

            sorted_teeth[i].quadrant = quadrant
            sorted_teeth[i].fdi_number = quadrant * 10 + current_pos

    assign_quadrant_with_gaps(q1_teeth, 1, upper_x_mid, gap_threshold_multiplier)
    assign_quadrant_with_gaps(q2_teeth, 2, upper_x_mid, gap_threshold_multiplier)
    assign_quadrant_with_gaps(q3_teeth, 3, lower_x_mid, gap_threshold_multiplier)
    assign_quadrant_with_gaps(q4_teeth, 4, lower_x_mid, gap_threshold_multiplier)

    return teeth


def detect_edentulous_zones(
    teeth: list[ToothComponent],
    image_shape: tuple[int, int],
    gap_threshold_multiplier: float = 1.5
) -> tuple[list[EdentulousZone], dict]:
    """
    Detect edentulous zones (gaps) between teeth.

    Args:
        teeth: List of teeth with FDI numbers assigned
        image_shape: (height, width) of the image
        gap_threshold_multiplier: Gap is detected if distance > multiplier * avg_tooth_width

    Returns:
        (list of EdentulousZone, quadrant_info dict)
    """
    zones = []
    quadrant_info = {}

    # Group teeth by quadrant
    quadrants = {1: [], 2: [], 3: [], 4: []}
    for tooth in teeth:
        if tooth.quadrant:
            quadrants[tooth.quadrant].append(tooth)

    for q_num, q_teeth in quadrants.items():
        # Sort by FDI position
        q_teeth_sorted = sorted(q_teeth, key=lambda t: t.fdi_number or 0)

        # Determine reference count for this quadrant based on highest FDI position
        if q_teeth_sorted:
            highest_pos = max((t.fdi_number or 0) % 10 for t in q_teeth_sorted)
            # If we see position 8, wisdom tooth is expected; otherwise assume 7
            reference = 8 if highest_pos >= 8 else 7
        else:
            reference = 7  # Default: assume no wisdom tooth

        # Calculate average tooth width in this quadrant
        if q_teeth_sorted:
            avg_width = np.mean([t.width for t in q_teeth_sorted])
            avg_height = np.mean([t.height for t in q_teeth_sorted])
        else:
            avg_width = 50  # Default fallback
            avg_height = 80

        quadrant_info[q_num] = {
            'detected': len(q_teeth_sorted),
            'reference': reference,
            'avg_width': avg_width,
            'avg_height': avg_height,
        }

        # Check for gaps between adjacent teeth using FDI numbers
        # Since FDI assignment now encodes gaps, non-consecutive FDI = missing teeth
        for i in range(len(q_teeth_sorted) - 1):
            t1 = q_teeth_sorted[i]
            t2 = q_teeth_sorted[i + 1]

            pos1 = (t1.fdi_number or 0) % 10
            pos2 = (t2.fdi_number or 0) % 10

            # If FDI positions are not consecutive, there's a gap
            if pos2 > pos1 + 1:
                missing_positions = list(range(pos1 + 1, pos2))
                missing_fdi = [q_num * 10 + p for p in missing_positions]
                missing_count = len(missing_positions)

                # Calculate zone bbox using actual bounding box edges (not centroid)
                # t1 has lower FDI (more anterior), t2 has higher FDI (more posterior)
                # bbox = (x, y, w, h) where x,y is top-left corner
                t1_bbox_left = t1.bbox[0]
                t1_bbox_right = t1.bbox[0] + t1.bbox[2]
                t2_bbox_left = t2.bbox[0]
                t2_bbox_right = t2.bbox[0] + t2.bbox[2]

                if q_num in [1, 4]:
                    # Image left side: lower FDI = higher X (toward midline)
                    # t1 (lower FDI) is at higher X, t2 (higher FDI) is at lower X
                    # Gap zone: from t2's right edge to t1's left edge
                    zone_x_start = t2_bbox_right  # Right edge of t2 (the posterior tooth)
                    zone_x_end = t1_bbox_left     # Left edge of t1 (the anterior tooth)
                else:
                    # Image right side: lower FDI = lower X (toward midline)
                    # t1 (lower FDI) is at lower X, t2 (higher FDI) is at higher X
                    # Gap zone: from t1's right edge to t2's left edge
                    zone_x_start = t1_bbox_right  # Right edge of t1 (the anterior tooth)
                    zone_x_end = t2_bbox_left     # Left edge of t2 (the posterior tooth)

                y_center = int((t1.centroid[1] + t2.centroid[1]) / 2)
                zone_height = int(avg_height)
                zone_width = zone_x_end - zone_x_start

                # Only add zone if it's meaningfully sized (at least 50% of avg tooth width)
                # and zone_width is positive (zone_x_end > zone_x_start)
                min_zone_width = avg_width * 0.5
                if zone_width >= min_zone_width:
                    zone = EdentulousZone(
                        quadrant=q_num,
                        missing_fdi_numbers=missing_fdi,
                        adjacent_teeth=(t1.fdi_number, t2.fdi_number),
                        bbox=(zone_x_start, y_center - zone_height // 2, zone_width, zone_height),
                        area=zone_width * zone_height,
                        missing_count=missing_count
                    )
                    zones.append(zone)

        # Check for missing teeth at the ends (positions 1 and 7/8)
        if q_teeth_sorted:
            first_pos = (q_teeth_sorted[0].fdi_number or 0) % 10
            last_pos = (q_teeth_sorted[-1].fdi_number or 0) % 10

            # Missing at anterior (position 1)
            if first_pos > 1:
                missing_anterior = list(range(1, first_pos))
                first_tooth = q_teeth_sorted[0]
                ft_left = first_tooth.bbox[0]
                ft_right = first_tooth.bbox[0] + first_tooth.bbox[2]

                # Estimate zone at anterior using actual bbox edges
                zone_width = int(avg_width * len(missing_anterior))
                if q_num in [1, 4]:
                    # Image left side: anterior is toward midline (right)
                    x_start = ft_right  # Right edge of first tooth
                    x_end = x_start + zone_width
                else:
                    # Image right side: anterior is toward midline (left)
                    x_end = ft_left  # Left edge of first tooth
                    x_start = x_end - zone_width

                y_center = int(first_tooth.centroid[1])
                zone = EdentulousZone(
                    quadrant=q_num,
                    missing_fdi_numbers=[q_num * 10 + p for p in missing_anterior],
                    adjacent_teeth=(None, first_tooth.fdi_number),
                    bbox=(min(x_start, x_end), y_center - int(avg_height) // 2,
                          abs(x_end - x_start), int(avg_height)),
                    area=abs(x_end - x_start) * int(avg_height),
                    missing_count=len(missing_anterior)
                )
                zones.append(zone)

            # Missing at posterior (toward position 7 or 8)
            if last_pos < reference:
                missing_posterior = list(range(last_pos + 1, reference + 1))
                last_tooth = q_teeth_sorted[-1]
                lt_left = last_tooth.bbox[0]
                lt_right = last_tooth.bbox[0] + last_tooth.bbox[2]

                # Estimate zone at posterior using actual bbox edges
                zone_width = int(avg_width * len(missing_posterior))
                if q_num in [1, 4]:
                    # Image left side: posterior is away from midline (left)
                    x_end = lt_left  # Left edge of last tooth
                    x_start = x_end - zone_width
                else:
                    # Image right side: posterior is away from midline (right)
                    x_start = lt_right  # Right edge of last tooth
                    x_end = x_start + zone_width

                y_center = int(last_tooth.centroid[1])
                zone = EdentulousZone(
                    quadrant=q_num,
                    missing_fdi_numbers=[q_num * 10 + p for p in missing_posterior],
                    adjacent_teeth=(last_tooth.fdi_number, None),
                    bbox=(min(x_start, x_end), y_center - int(avg_height) // 2,
                          abs(x_end - x_start), int(avg_height)),
                    area=abs(x_end - x_start) * int(avg_height),
                    missing_count=len(missing_posterior)
                )
                zones.append(zone)

    return zones, quadrant_info


def analyze_dental_panorama(
    prediction_mask: np.ndarray,
    image_shape: tuple[int, int] | None = None,
    erode_iterations: int = 2,
    open_iterations: int = 3,
    min_tooth_area: int = 2000,
    gap_threshold: float = 1.5
) -> DentalAnalysisResult:
    """
    Main entry point for dental panorama analysis.

    Args:
        prediction_mask: BGR prediction image from the model
        image_shape: (height, width) - if None, derived from prediction_mask
        erode_iterations: Erosion iterations for CCA
        open_iterations: Opening iterations for CCA
        min_tooth_area: Minimum area to consider as tooth
        gap_threshold: Multiplier for gap detection threshold

    Returns:
        DentalAnalysisResult with all analysis data
    """
    if image_shape is None:
        image_shape = prediction_mask.shape[:2]

    # Extract tooth components
    teeth = extract_tooth_components(
        prediction_mask,
        erode_iterations=erode_iterations,
        open_iterations=open_iterations,
        min_area=min_tooth_area
    )

    # Assign FDI numbers (uses gap_threshold for spacing-based gap detection)
    teeth = assign_fdi_numbers(teeth, image_shape, gap_threshold_multiplier=gap_threshold)

    # Detect edentulous zones
    zones, quadrant_info = detect_edentulous_zones(teeth, image_shape, gap_threshold)

    # Calculate totals
    total_detected = len(teeth)
    total_missing = sum(z.missing_count for z in zones)

    # Determine reference count (sum of per-quadrant references)
    reference_count = sum(qi['reference'] for qi in quadrant_info.values())

    return DentalAnalysisResult(
        detected_teeth=teeth,
        edentulous_zones=zones,
        reference_count=reference_count,
        total_detected=total_detected,
        total_missing=total_missing,
        quadrant_info=quadrant_info
    )


def visualize_analysis(
    original_image: np.ndarray,
    result: DentalAnalysisResult,
    show_fdi_labels: bool = True,
    show_zones: bool = True,
    show_measurements: bool = True
) -> np.ndarray:
    """
    Create visualization of the dental analysis.

    Args:
        original_image: Original X-ray image (BGR)
        result: DentalAnalysisResult from analyze_dental_panorama
        show_fdi_labels: Draw FDI numbers on each tooth
        show_zones: Highlight edentulous zones
        show_measurements: Show gap measurements

    Returns:
        Annotated image (BGR)
    """
    image = original_image.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw teeth with FDI labels
    if show_fdi_labels:
        for tooth in result.detected_teeth:
            # Draw contour
            color = (0, 255, 0)  # Green for detected teeth
            cv2.drawContours(image, [tooth.contour], 0, color, 2)

            # Draw FDI label
            if tooth.fdi_number:
                cx, cy = int(tooth.centroid[0]), int(tooth.centroid[1])
                label = str(tooth.fdi_number)

                # Background rectangle for label
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image,
                              (cx - text_w // 2 - 2, cy - text_h - 2),
                              (cx + text_w // 2 + 2, cy + 2),
                              (0, 0, 0), -1)
                cv2.putText(image, label,
                           (cx - text_w // 2, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw edentulous zones
    if show_zones:
        for zone in result.edentulous_zones:
            x, y, w, h = zone.bbox
            if w > 0 and h > 0:
                # Red rectangle for missing teeth zone
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Hatched fill pattern (diagonal lines)
                for i in range(0, w + h, 10):
                    pt1 = (x + min(i, w), y + max(0, i - w))
                    pt2 = (x + max(0, i - h), y + min(i, h))
                    cv2.line(image, pt1, pt2, (0, 0, 255), 1)

                if show_measurements:
                    # Label with missing FDI numbers
                    missing_str = ",".join(str(f) for f in zone.missing_fdi_numbers[:3])
                    if len(zone.missing_fdi_numbers) > 3:
                        missing_str += "..."

                    label = f"Missing: {missing_str}"
                    cv2.putText(image, label, (x, y - 15),
                               cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

                    # Dimensions
                    dim_label = f"{w}x{h}px"
                    cv2.putText(image, dim_label, (x, y + h + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Summary text at top
    summary = f"Detected: {result.total_detected} | Missing: {result.total_missing} | Reference: {result.reference_count}"
    cv2.rectangle(image, (5, 5), (500, 30), (0, 0, 0), -1)
    cv2.putText(image, summary, (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image


def print_analysis_report(result: DentalAnalysisResult) -> str:
    """Generate a text report of the analysis."""
    lines = [
        "=" * 60,
        "DENTAL PANORAMA ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Total teeth detected: {result.total_detected}",
        f"Total missing teeth: {result.total_missing}",
        f"Reference count: {result.reference_count}",
        "",
        "QUADRANT BREAKDOWN:",
        "-" * 40,
    ]

    quadrant_names = {
        1: "Q1 (Upper Right)",
        2: "Q2 (Upper Left)",
        3: "Q3 (Lower Left)",
        4: "Q4 (Lower Right)"
    }

    for q_num in [1, 2, 3, 4]:
        if q_num in result.quadrant_info:
            qi = result.quadrant_info[q_num]
            lines.append(f"  {quadrant_names[q_num]}:")
            lines.append(f"    Detected: {qi['detected']} / {qi['reference']}")
            lines.append(f"    Avg tooth width: {qi['avg_width']:.1f}px")

    if result.edentulous_zones:
        lines.extend([
            "",
            "EDENTULOUS ZONES:",
            "-" * 40,
        ])
        for i, zone in enumerate(result.edentulous_zones, 1):
            adj_str = f"{zone.adjacent_teeth[0] or 'edge'} - {zone.adjacent_teeth[1] or 'edge'}"
            lines.append(f"  Zone {i} (Q{zone.quadrant}):")
            lines.append(f"    Missing FDI: {zone.missing_fdi_numbers}")
            lines.append(f"    Between: {adj_str}")
            lines.append(f"    Size: {zone.bbox[2]}x{zone.bbox[3]}px ({zone.area}px^2)")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return report


def convert_zones_to_physical(
    result: DentalAnalysisResult,
    priors: PanoMetricPriors,
) -> list[EdentulousZonePhysical]:
    """
    Convert edentulous zones to physical (mm) measurements.

    Args:
        result: DentalAnalysisResult from analyze_dental_panorama
        priors: Calibration priors from physical_metrics module

    Returns:
        List of EdentulousZonePhysical with mm measurements and uncertainty

    Example:
        >>> from physical_metrics import create_priors_from_image_dimensions
        >>> priors = create_priors_from_image_dimensions(3126, 1300, fov_width_mm=270)
        >>> zones_mm = convert_zones_to_physical(analysis_result, priors)
        >>> for zone in zones_mm:
        ...     print(f"Zone width: {zone.width}")
    """
    from physical_metrics import measure_edentulous_zone_physical

    return [
        measure_edentulous_zone_physical(zone, priors)
        for zone in result.edentulous_zones
    ]


def visualize_analysis_physical(
    original_image: np.ndarray,
    result: DentalAnalysisResult,
    priors: PanoMetricPriors,
    show_fdi_labels: bool = True,
    show_zones: bool = True,
    show_measurements: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Create visualization with physical (mm) measurements.

    Args:
        original_image: Original X-ray image (BGR)
        result: DentalAnalysisResult from analyze_dental_panorama
        priors: Calibration priors for mm conversion
        show_fdi_labels: Draw FDI numbers on each tooth
        show_zones: Highlight edentulous zones
        show_measurements: Show physical measurements
        show_confidence: Show confidence level indicators

    Returns:
        Annotated image (BGR) with physical measurements
    """
    from physical_metrics import measure_edentulous_zone_physical, ConfidenceLevel

    image = original_image.copy()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw teeth with FDI labels
    if show_fdi_labels:
        for tooth in result.detected_teeth:
            color = (0, 255, 0)  # Green for detected teeth
            cv2.drawContours(image, [tooth.contour], 0, color, 2)

            if tooth.fdi_number:
                cx, cy = int(tooth.centroid[0]), int(tooth.centroid[1])
                label = str(tooth.fdi_number)

                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image,
                              (cx - text_w // 2 - 2, cy - text_h - 2),
                              (cx + text_w // 2 + 2, cy + 2),
                              (0, 0, 0), -1)
                cv2.putText(image, label,
                           (cx - text_w // 2, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw edentulous zones with physical measurements
    if show_zones:
        for zone in result.edentulous_zones:
            zone_physical = measure_edentulous_zone_physical(zone, priors)

            x, y, w, h = zone.bbox
            if w > 0 and h > 0:
                # Color based on confidence level
                if show_confidence:
                    confidence = zone_physical.width.confidence
                    if confidence == ConfidenceLevel.HIGH:
                        color = (0, 200, 0)    # Green - high confidence
                    elif confidence == ConfidenceLevel.MEDIUM:
                        color = (0, 165, 255)  # Orange - medium
                    elif confidence == ConfidenceLevel.LOW:
                        color = (0, 100, 255)  # Red-orange - low
                    else:
                        color = (0, 0, 255)    # Red - very low
                else:
                    color = (0, 0, 255)  # Red

                # Draw rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                # Hatched fill pattern
                for i in range(0, w + h, 10):
                    pt1 = (x + min(i, w), y + max(0, i - w))
                    pt2 = (x + max(0, i - h), y + min(i, h))
                    cv2.line(image, pt1, pt2, color, 1)

                if show_measurements:
                    # Missing FDI numbers
                    missing_str = ",".join(str(f) for f in zone.missing_fdi_numbers[:3])
                    if len(zone.missing_fdi_numbers) > 3:
                        missing_str += "..."

                    # Physical dimensions
                    width_mm = zone_physical.width.value_mm
                    height_mm = zone_physical.height.value_mm
                    uncertainty = zone_physical.width.uncertainty_percent

                    # Label with physical measurements
                    label1 = f"Missing: {missing_str}"
                    label2 = f"{width_mm:.1f}x{height_mm:.1f}mm"

                    if show_confidence:
                        conf_str = zone_physical.width.confidence.value[0].upper()  # H/M/L/V
                        label2 += f" [{conf_str}]"

                    cv2.putText(image, label1, (x, y - 50),
                               cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                    cv2.putText(image, label2, (x, y - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    # Range below
                    range_str = f"({zone_physical.width.min_mm:.1f}-{zone_physical.width.max_mm:.1f}mm)"
                    cv2.putText(image, range_str, (x, y + h + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Summary text at top
    calibration_note = "CALIBRATED" if priors.is_calibrated() else "UNCALIBRATED"
    summary = f"Detected: {result.total_detected} | Missing: {result.total_missing} | {calibration_note}"
    cv2.rectangle(image, (5, 5), (600, 30), (0, 0, 0), -1)
    cv2.putText(image, summary, (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Legend for confidence colors
    if show_confidence and show_zones:
        legend_y = 50
        cv2.rectangle(image, (5, legend_y), (200, legend_y + 80), (0, 0, 0), -1)
        cv2.putText(image, "Confidence:", (10, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(image, (10, legend_y + 22), (20, legend_y + 32), (0, 200, 0), -1)
        cv2.putText(image, "High", (25, legend_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.rectangle(image, (60, legend_y + 22), (70, legend_y + 32), (0, 165, 255), -1)
        cv2.putText(image, "Med", (75, legend_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.rectangle(image, (110, legend_y + 22), (120, legend_y + 32), (0, 100, 255), -1)
        cv2.putText(image, "Low", (125, legend_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.rectangle(image, (160, legend_y + 22), (170, legend_y + 32), (0, 0, 255), -1)
        cv2.putText(image, "V.Low", (175, legend_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Warning for uncalibrated
        if not priors.is_calibrated():
            cv2.putText(image, "! Uncalibrated - verify with CBCT", (10, legend_y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1)

    return image
