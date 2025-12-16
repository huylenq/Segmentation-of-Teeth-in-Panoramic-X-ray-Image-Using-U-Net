"""
Physical Metrics Module for Panoramic X-ray Measurements.

This module provides calibration and conversion utilities for converting
pixel measurements to physical (mm) measurements from panoramic X-ray images.

IMPORTANT LIMITATIONS (from peer-reviewed literature):
- Panoramic radiography has spatially-varying magnification (15-30%)
- Horizontal measurements are LESS reliable than vertical
- Anterior region has HIGHEST uncertainty
- Without internal calibration, errors range 0.5-7.5mm (mean ~3mm)
- For clinical decisions requiring <1mm precision, use CBCT

References:
- Devlin H, Yuan J (2013) - Object position and image magnification
- Welander U et al (1990) - Mathematical theory of rotational panoramic radiography
- Yeom HG et al (2018) - Ball phantom for focal trough evaluation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
import numpy as np


class DentalRegion(Enum):
    """Dental arch regions with different magnification characteristics."""
    ANTERIOR = "anterior"      # Incisors (FDI positions 1-2)
    CANINE_PREMOLAR = "canine_premolar"  # Canines and premolars (3-5)
    MOLAR = "molar"           # Molars (6-8)


class ConfidenceLevel(Enum):
    """Confidence level for physical measurements."""
    HIGH = "high"       # ±5% uncertainty (calibrated, molar region)
    MEDIUM = "medium"   # ±10% uncertainty (calibrated, premolar)
    LOW = "low"         # ±15-20% uncertainty (uncalibrated or anterior)
    VERY_LOW = "very_low"  # >20% uncertainty (should use CBCT)


@dataclass
class MagnificationFactors:
    """
    Regional magnification factors for panoramic X-ray.

    These are MACHINE-SPECIFIC and should ideally be calibrated
    using a phantom for each X-ray unit.

    Literature defaults for typical OPG machines:
    - Vertical MF: 1.25-1.28 (relatively consistent)
    - Horizontal MF: 1.20-1.32 (varies by region and positioning)
    """
    # Vertical magnification (more consistent across regions)
    vertical: float = 1.27

    # Horizontal magnification by region (less reliable)
    horizontal_anterior: float = 1.28      # Highest variability
    horizontal_canine_premolar: float = 1.25
    horizontal_molar: float = 1.22         # Most reliable

    # Uncertainty bounds (as fraction, e.g., 0.05 = ±5%)
    vertical_uncertainty: float = 0.03
    horizontal_anterior_uncertainty: float = 0.15  # Very high
    horizontal_canine_premolar_uncertainty: float = 0.08
    horizontal_molar_uncertainty: float = 0.05

    def get_horizontal_mf(self, region: DentalRegion) -> tuple[float, float]:
        """Get horizontal MF and uncertainty for a region."""
        if region == DentalRegion.ANTERIOR:
            return self.horizontal_anterior, self.horizontal_anterior_uncertainty
        elif region == DentalRegion.CANINE_PREMOLAR:
            return self.horizontal_canine_premolar, self.horizontal_canine_premolar_uncertainty
        else:
            return self.horizontal_molar, self.horizontal_molar_uncertainty


# Default magnification factors based on literature
# These are APPROXIMATE - should be calibrated per machine
X800_DEFAULT_MF = MagnificationFactors(
    vertical=1.27,
    horizontal_anterior=1.28,
    horizontal_canine_premolar=1.25,
    horizontal_molar=1.22,
    vertical_uncertainty=0.03,
    horizontal_anterior_uncertainty=0.15,
    horizontal_canine_premolar_uncertainty=0.10,
    horizontal_molar_uncertainty=0.06,
)


@dataclass
class CalibrationReference:
    """
    Internal calibration reference object (e.g., metal ball).

    This is the GOLD STANDARD for panoramic measurements.
    A 5-6mm metal ball placed in the region of interest allows
    computing local magnification factor.
    """
    true_size_mm: float           # Known physical size (e.g., 6.0mm)
    measured_size_px: float       # Measured size in image pixels
    region: DentalRegion          # Where the reference was placed
    is_horizontal: bool = True    # Measured horizontally or vertically

    def compute_local_mf(self, pixel_spacing_mm: float) -> float:
        """Compute local magnification factor from reference."""
        measured_mm = self.measured_size_px * pixel_spacing_mm
        return measured_mm / self.true_size_mm


@dataclass
class PanoMetricPriors:
    """
    Calibration priors required for physical measurements from PNG images.

    Since PNG images lack DICOM metadata, these must be provided at inference time.

    There are several ways to establish pixel_spacing_mm:
    1. From machine specs: image_physical_width_mm / image_width_px
    2. From reference object: calibration_reference
    3. From average tooth size estimation (least accurate)
    """
    # === PRIMARY CALIBRATION (one of these is required) ===

    # Option 1: Known pixel spacing (mm per pixel at detector plane)
    # For X800 standard panoramic: typical image ~2800-3200px wide, ~150mm FOV
    pixel_spacing_mm: float | None = None

    # Option 2: Internal calibration object (most accurate)
    calibration_reference: CalibrationReference | None = None

    # Option 3: Estimate from image dimensions and assumed FOV
    # X800 panoramic: approximately 240-300mm horizontal FOV
    image_width_px: int | None = None
    assumed_fov_width_mm: float | None = None

    # === MAGNIFICATION FACTORS ===
    magnification_factors: MagnificationFactors = field(
        default_factory=lambda: X800_DEFAULT_MF
    )

    # === METADATA ===
    machine_model: str = "Morita Veraview X800"
    arch_form: Literal["standard", "narrow", "wide"] = "standard"
    calibration_method: str = "uncalibrated"

    def __post_init__(self):
        """Validate and derive pixel spacing if possible."""
        if self.pixel_spacing_mm is None:
            if self.image_width_px and self.assumed_fov_width_mm:
                self.pixel_spacing_mm = self.assumed_fov_width_mm / self.image_width_px
                self.calibration_method = "assumed_fov"
            elif self.calibration_reference:
                # Will be computed when convert_to_mm is called
                self.calibration_method = "reference_object"
        else:
            if self.calibration_reference:
                self.calibration_method = "reference_object"
            else:
                self.calibration_method = "manual_pixel_spacing"

    def get_effective_pixel_spacing(self) -> float:
        """Get the pixel spacing to use for conversions."""
        if self.pixel_spacing_mm is not None:
            return self.pixel_spacing_mm
        elif self.image_width_px and self.assumed_fov_width_mm:
            return self.assumed_fov_width_mm / self.image_width_px
        else:
            raise ValueError(
                "Cannot determine pixel spacing. Provide one of:\n"
                "1. pixel_spacing_mm directly\n"
                "2. image_width_px + assumed_fov_width_mm\n"
                "3. calibration_reference with pixel_spacing_mm"
            )

    def is_calibrated(self) -> bool:
        """Check if measurements will use calibration reference."""
        return self.calibration_reference is not None


@dataclass
class PhysicalMeasurement:
    """A physical measurement with uncertainty bounds."""
    value_mm: float                      # Best estimate in mm
    min_mm: float                        # Lower bound (pessimistic)
    max_mm: float                        # Upper bound (optimistic)
    uncertainty_percent: float           # Uncertainty as percentage
    confidence: ConfidenceLevel          # Qualitative confidence
    region: DentalRegion                 # Which dental region
    measurement_type: Literal["width", "height", "area"]

    # Recommendations
    clinical_note: str | None = None     # Clinical guidance

    def __str__(self) -> str:
        return f"{self.value_mm:.1f}mm (±{self.uncertainty_percent:.0f}%, {self.confidence.value})"

    def as_range_str(self) -> str:
        return f"{self.min_mm:.1f}-{self.max_mm:.1f}mm"


@dataclass
class EdentulousZonePhysical:
    """Physical measurements for an edentulous zone."""
    # Pixel measurements (original)
    width_px: int
    height_px: int
    area_px: int

    # Physical measurements
    width: PhysicalMeasurement
    height: PhysicalMeasurement
    area_mm2: float                      # Best estimate area
    area_range_mm2: tuple[float, float]  # (min, max) area

    # Zone identification
    quadrant: int
    missing_fdi_numbers: list[int]
    adjacent_teeth: tuple[int | None, int | None]

    # Clinical guidance
    implant_feasibility: str | None = None
    recommendation: str | None = None


def classify_fdi_region(fdi_number: int) -> DentalRegion:
    """
    Classify a tooth by FDI number into dental region.

    FDI positions within quadrant:
    - 1-2: Incisors (anterior)
    - 3-5: Canines and premolars
    - 6-8: Molars
    """
    position = fdi_number % 10
    if position <= 2:
        return DentalRegion.ANTERIOR
    elif position <= 5:
        return DentalRegion.CANINE_PREMOLAR
    else:
        return DentalRegion.MOLAR


def classify_zone_region(missing_fdi_numbers: list[int]) -> DentalRegion:
    """
    Classify an edentulous zone's region based on missing teeth.

    Uses the most anterior (lowest position) tooth to determine region,
    as it has the highest uncertainty.
    """
    if not missing_fdi_numbers:
        return DentalRegion.MOLAR  # Default

    positions = [fdi % 10 for fdi in missing_fdi_numbers]
    min_position = min(positions)

    if min_position <= 2:
        return DentalRegion.ANTERIOR
    elif min_position <= 5:
        return DentalRegion.CANINE_PREMOLAR
    else:
        return DentalRegion.MOLAR


def get_confidence_level(
    region: DentalRegion,
    is_calibrated: bool,
    measurement_type: Literal["width", "height"]
) -> ConfidenceLevel:
    """Determine confidence level based on region and calibration."""
    if measurement_type == "height":
        # Vertical measurements are more reliable
        if is_calibrated:
            return ConfidenceLevel.HIGH
        elif region == DentalRegion.MOLAR:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    else:
        # Horizontal measurements (width) are less reliable
        if is_calibrated and region == DentalRegion.MOLAR:
            return ConfidenceLevel.HIGH
        elif is_calibrated:
            return ConfidenceLevel.MEDIUM
        elif region == DentalRegion.MOLAR:
            return ConfidenceLevel.MEDIUM
        elif region == DentalRegion.CANINE_PREMOLAR:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


def convert_to_physical(
    width_px: int,
    height_px: int,
    priors: PanoMetricPriors,
    zone_region: DentalRegion,
) -> tuple[PhysicalMeasurement, PhysicalMeasurement]:
    """
    Convert pixel dimensions to physical measurements with uncertainty.

    Args:
        width_px: Horizontal extent in pixels
        height_px: Vertical extent in pixels
        priors: Calibration priors
        zone_region: Which dental region this measurement is in

    Returns:
        (width_measurement, height_measurement)
    """
    px_spacing = priors.get_effective_pixel_spacing()
    mf = priors.magnification_factors
    is_calibrated = priors.is_calibrated()

    # Get magnification factors for this region
    h_mf, h_uncertainty = mf.get_horizontal_mf(zone_region)
    v_mf = mf.vertical
    v_uncertainty = mf.vertical_uncertainty

    # If we have a calibration reference, use it to refine MF
    if priors.calibration_reference:
        ref = priors.calibration_reference
        local_mf = ref.compute_local_mf(px_spacing)
        if ref.is_horizontal:
            # Use reference to correct horizontal MF
            h_mf = local_mf
            h_uncertainty = 0.05  # Reduced uncertainty with calibration
        else:
            v_mf = local_mf
            v_uncertainty = 0.03

    # Convert width (horizontal)
    width_raw_mm = width_px * px_spacing
    width_corrected_mm = width_raw_mm / h_mf
    width_min = width_corrected_mm * (1 - h_uncertainty)
    width_max = width_corrected_mm * (1 + h_uncertainty)

    width_measurement = PhysicalMeasurement(
        value_mm=width_corrected_mm,
        min_mm=width_min,
        max_mm=width_max,
        uncertainty_percent=h_uncertainty * 100,
        confidence=get_confidence_level(zone_region, is_calibrated, "width"),
        region=zone_region,
        measurement_type="width",
        clinical_note=_get_width_clinical_note(zone_region, is_calibrated),
    )

    # Convert height (vertical)
    height_raw_mm = height_px * px_spacing
    height_corrected_mm = height_raw_mm / v_mf
    height_min = height_corrected_mm * (1 - v_uncertainty)
    height_max = height_corrected_mm * (1 + v_uncertainty)

    height_measurement = PhysicalMeasurement(
        value_mm=height_corrected_mm,
        min_mm=height_min,
        max_mm=height_max,
        uncertainty_percent=v_uncertainty * 100,
        confidence=get_confidence_level(zone_region, is_calibrated, "height"),
        region=zone_region,
        measurement_type="height",
        clinical_note=_get_height_clinical_note(zone_region, is_calibrated),
    )

    return width_measurement, height_measurement


def _get_width_clinical_note(region: DentalRegion, is_calibrated: bool) -> str:
    """Get clinical guidance note for width measurement."""
    if region == DentalRegion.ANTERIOR:
        if not is_calibrated:
            return "CAUTION: Anterior horizontal measurements are unreliable. CBCT recommended for implant planning."
        return "Anterior region: verify with CBCT for critical decisions."
    elif region == DentalRegion.CANINE_PREMOLAR:
        if not is_calibrated:
            return "Premolar region: moderate uncertainty. Consider CBCT verification."
        return "Calibrated premolar measurement."
    else:
        return "Molar region: most reliable horizontal measurement."


def _get_height_clinical_note(region: DentalRegion, is_calibrated: bool) -> str:
    """Get clinical guidance note for height measurement."""
    if is_calibrated:
        return "Vertical measurement with calibration reference."
    return "Vertical measurements are generally more reliable than horizontal."


def measure_edentulous_zone_physical(
    zone,  # EdentulousZone from tooth_analysis
    priors: PanoMetricPriors,
) -> EdentulousZonePhysical:
    """
    Convert an EdentulousZone to physical measurements.

    Args:
        zone: EdentulousZone from tooth_analysis module
        priors: Calibration priors for the image

    Returns:
        EdentulousZonePhysical with mm measurements and uncertainty
    """
    # Determine region from missing teeth
    region = classify_zone_region(zone.missing_fdi_numbers)

    # Get pixel dimensions from bbox
    width_px = zone.bbox[2]
    height_px = zone.bbox[3]
    area_px = zone.area

    # Convert to physical
    width_mm, height_mm = convert_to_physical(
        width_px, height_px, priors, region
    )

    # Calculate area (use best estimates, propagate uncertainty)
    area_best = width_mm.value_mm * height_mm.value_mm
    area_min = width_mm.min_mm * height_mm.min_mm
    area_max = width_mm.max_mm * height_mm.max_mm

    # Generate implant feasibility assessment
    implant_note = _assess_implant_feasibility(width_mm, height_mm, region)

    # Generate recommendation
    recommendation = _generate_recommendation(width_mm, height_mm, region, priors.is_calibrated())

    return EdentulousZonePhysical(
        width_px=width_px,
        height_px=height_px,
        area_px=area_px,
        width=width_mm,
        height=height_mm,
        area_mm2=area_best,
        area_range_mm2=(area_min, area_max),
        quadrant=zone.quadrant,
        missing_fdi_numbers=zone.missing_fdi_numbers,
        adjacent_teeth=zone.adjacent_teeth,
        implant_feasibility=implant_note,
        recommendation=recommendation,
    )


def _assess_implant_feasibility(
    width: PhysicalMeasurement,
    height: PhysicalMeasurement,
    region: DentalRegion,
) -> str:
    """
    Assess implant feasibility based on zone dimensions.

    Standard implant diameters: 3.5-5.0mm
    Minimum bone requirements vary by region.
    """
    # Minimum widths for single implant (conservative estimates)
    # These account for 1.5-2mm safety margin on each side
    min_width_single = 6.0  # mm (for 3.5mm implant + margins)
    min_width_standard = 7.0  # mm (for 4.0mm implant + margins)

    w = width.value_mm
    w_min = width.min_mm

    if w_min >= min_width_standard:
        return f"Width ({w:.1f}mm) likely sufficient for standard implant"
    elif w_min >= min_width_single:
        return f"Width ({w:.1f}mm) may accommodate narrow implant; verify with CBCT"
    elif w >= min_width_single:
        return f"Width ({w:.1f}mm) borderline; uncertainty range suggests CBCT verification needed"
    else:
        return f"Width ({w:.1f}mm) appears insufficient for single implant"


def _generate_recommendation(
    width: PhysicalMeasurement,
    height: PhysicalMeasurement,
    region: DentalRegion,
    is_calibrated: bool,
) -> str:
    """Generate clinical recommendation based on measurements."""
    recommendations = []

    # Confidence-based recommendations
    if width.confidence == ConfidenceLevel.VERY_LOW:
        recommendations.append("LOW CONFIDENCE: Horizontal measurement unreliable. CBCT required for clinical decisions.")
    elif width.confidence == ConfidenceLevel.LOW:
        recommendations.append("MODERATE UNCERTAINTY: Consider CBCT verification before treatment planning.")

    # Region-specific recommendations
    if region == DentalRegion.ANTERIOR:
        recommendations.append("Anterior region: 83% of panoramic measurements underestimate actual dimensions.")

    # Calibration recommendation
    if not is_calibrated:
        recommendations.append("For improved accuracy: place 5-6mm reference ball at site of interest.")

    return " ".join(recommendations) if recommendations else "Measurements within acceptable uncertainty range."


# === Convenience functions for common calibration scenarios ===

def create_priors_from_image_dimensions(
    image_width_px: int,
    image_height_px: int,
    fov_width_mm: float = 270.0,  # Typical X800 panoramic FOV
    machine: str = "X800",
) -> PanoMetricPriors:
    """
    Create calibration priors from image dimensions and assumed FOV.

    This is the LEAST accurate method but works when no other
    calibration information is available.

    Args:
        image_width_px: Image width in pixels
        image_height_px: Image height in pixels
        fov_width_mm: Assumed horizontal field of view (default 270mm for X800)
        machine: Machine identifier

    Returns:
        PanoMetricPriors configured for the image
    """
    return PanoMetricPriors(
        image_width_px=image_width_px,
        assumed_fov_width_mm=fov_width_mm,
        machine_model=f"Morita Veraview {machine} (estimated FOV)",
    )


def create_priors_from_reference_tooth(
    image_width_px: int,
    reference_tooth_width_px: float,
    reference_fdi: int,
    estimated_tooth_width_mm: float | None = None,
) -> PanoMetricPriors:
    """
    Create calibration priors using a detected tooth as reference.

    Uses population average tooth widths as reference:
    - Central incisor: ~8.5mm
    - Lateral incisor: ~6.5mm
    - Canine: ~7.5mm
    - Premolars: ~7.0mm
    - Molars: ~10.0-11.0mm

    This is MORE accurate than FOV estimation but LESS accurate
    than using a known calibration object.

    Args:
        image_width_px: Image width in pixels
        reference_tooth_width_px: Width of reference tooth in pixels
        reference_fdi: FDI number of reference tooth
        estimated_tooth_width_mm: Override for tooth width (if known)

    Returns:
        PanoMetricPriors configured for the image
    """
    # Population average mesiodistal widths (mm)
    # Source: Wheeler's Dental Anatomy
    AVERAGE_TOOTH_WIDTHS = {
        # Maxillary (quadrants 1, 2)
        11: 8.5, 21: 8.5,  # Central incisors
        12: 6.5, 22: 6.5,  # Lateral incisors
        13: 7.5, 23: 7.5,  # Canines
        14: 7.0, 24: 7.0,  # First premolars
        15: 6.5, 25: 6.5,  # Second premolars
        16: 10.0, 26: 10.0,  # First molars
        17: 9.0, 27: 9.0,  # Second molars
        18: 8.5, 28: 8.5,  # Third molars
        # Mandibular (quadrants 3, 4)
        31: 5.5, 41: 5.5,  # Central incisors
        32: 6.0, 42: 6.0,  # Lateral incisors
        33: 7.0, 43: 7.0,  # Canines
        34: 7.0, 44: 7.0,  # First premolars
        35: 7.0, 45: 7.0,  # Second premolars
        36: 11.0, 46: 11.0,  # First molars
        37: 10.5, 47: 10.5,  # Second molars
        38: 10.0, 48: 10.0,  # Third molars
    }

    if estimated_tooth_width_mm is None:
        if reference_fdi not in AVERAGE_TOOTH_WIDTHS:
            raise ValueError(f"Unknown FDI number: {reference_fdi}")
        estimated_tooth_width_mm = AVERAGE_TOOTH_WIDTHS[reference_fdi]

    # Calculate pixel spacing from reference tooth
    # Note: This includes magnification, so it's not true detector pixel spacing
    # but it's the effective spacing for this region
    region = classify_fdi_region(reference_fdi)
    mf = X800_DEFAULT_MF
    h_mf, _ = mf.get_horizontal_mf(region)

    # pixel_spacing = (tooth_width_mm * magnification) / tooth_width_px
    effective_spacing = (estimated_tooth_width_mm * h_mf) / reference_tooth_width_px

    return PanoMetricPriors(
        pixel_spacing_mm=effective_spacing,
        machine_model="Morita Veraview X800 (tooth reference calibration)",
        calibration_method="tooth_reference",
    )


def create_priors_from_calibration_ball(
    image_width_px: int,
    ball_diameter_mm: float,
    ball_measured_px: float,
    ball_region: DentalRegion,
    ball_is_horizontal: bool = True,
) -> PanoMetricPriors:
    """
    Create calibration priors using an internal calibration ball.

    This is the GOLD STANDARD for panoramic measurements.

    Args:
        image_width_px: Image width in pixels
        ball_diameter_mm: Known physical diameter of ball (typically 5-6mm)
        ball_measured_px: Measured diameter of ball in image pixels
        ball_region: Which dental region the ball was placed in
        ball_is_horizontal: Whether measurement is horizontal (True) or vertical (False)

    Returns:
        PanoMetricPriors configured with calibration reference
    """
    # Estimate pixel spacing from ball
    # pixel_spacing = ball_diameter_mm / ball_measured_px (approximate)
    pixel_spacing = ball_diameter_mm / ball_measured_px

    calibration = CalibrationReference(
        true_size_mm=ball_diameter_mm,
        measured_size_px=ball_measured_px,
        region=ball_region,
        is_horizontal=ball_is_horizontal,
    )

    return PanoMetricPriors(
        pixel_spacing_mm=pixel_spacing,
        calibration_reference=calibration,
        machine_model="Morita Veraview X800 (ball calibration)",
    )


def print_physical_measurement_report(
    zones_physical: list[EdentulousZonePhysical],
    priors: PanoMetricPriors,
) -> str:
    """
    Generate a detailed report of physical measurements.

    Args:
        zones_physical: List of EdentulousZonePhysical measurements
        priors: Calibration priors used

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        "EDENTULOUS ZONE PHYSICAL MEASUREMENTS",
        "=" * 70,
        "",
        "CALIBRATION INFO:",
        f"  Machine: {priors.machine_model}",
        f"  Method: {priors.calibration_method}",
        f"  Pixel spacing: {priors.get_effective_pixel_spacing():.4f} mm/px",
        f"  Calibrated: {'Yes' if priors.is_calibrated() else 'No'}",
        "",
    ]

    if not priors.is_calibrated():
        lines.extend([
            "  ⚠️  WARNING: Measurements are UNCALIBRATED",
            "  For clinical accuracy, use 5-6mm calibration ball or CBCT",
            "",
        ])

    lines.extend([
        "MEASUREMENTS:",
        "-" * 50,
    ])

    for i, zone in enumerate(zones_physical, 1):
        adj_str = f"{zone.adjacent_teeth[0] or 'edge'} - {zone.adjacent_teeth[1] or 'edge'}"
        lines.extend([
            f"",
            f"Zone {i} (Q{zone.quadrant}): Missing {zone.missing_fdi_numbers}",
            f"  Between: {adj_str}",
            f"",
            f"  Width:  {zone.width}",
            f"          Range: {zone.width.as_range_str()}",
            f"          Confidence: {zone.width.confidence.value.upper()}",
            f"",
            f"  Height: {zone.height}",
            f"          Range: {zone.height.as_range_str()}",
            f"          Confidence: {zone.height.confidence.value.upper()}",
            f"",
            f"  Area:   {zone.area_mm2:.1f} mm² ({zone.area_range_mm2[0]:.1f}-{zone.area_range_mm2[1]:.1f} mm²)",
            f"",
            f"  Implant: {zone.implant_feasibility}",
        ])

        if zone.recommendation:
            # Wrap long recommendations
            rec_lines = zone.recommendation.split(". ")
            lines.append(f"  Note: {rec_lines[0]}.")
            for rec in rec_lines[1:]:
                if rec.strip():
                    lines.append(f"        {rec}.")

    lines.extend([
        "",
        "=" * 70,
        "DISCLAIMER: Panoramic measurements have inherent limitations.",
        "For implant planning, CBCT verification is strongly recommended.",
        "=" * 70,
    ])

    report = "\n".join(lines)
    print(report)
    return report
