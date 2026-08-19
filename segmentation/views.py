import base64
import cv2
import numpy as np
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

from .inference import VesselSegmenter


segmenter = VesselSegmenter()


def image_to_base64(image):
    ok, encoded = cv2.imencode(".png", image)

    if not ok:
        return None

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


@ensure_csrf_cookie
def home(request):

    if request.method == "POST":

        uploaded_file = request.FILES.get("retinal_image")

        if not uploaded_file:
            return render(
                request,
                "segmentation/home.html",
                {"error": "Please select a retinal image."}
            )

        data = np.frombuffer(
            uploaded_file.read(),
            np.uint8
        )

        image = cv2.imdecode(
            data,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return render(
                request,
                "segmentation/home.html",
                {"error": "Could not read the uploaded image."}
            )

        # Run trained U-Net
        prediction, probability_map = segmenter.predict(
            image,
            return_probability=True
        )

        # Image statistics
        height, width = image.shape[:2]

        gray_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        vessel_pixels = int((prediction > 0).sum())
        total_pixels = int(prediction.size)
        background_pixels = total_pixels - vessel_pixels

        vessel_coverage = round(
            float(vessel_pixels / total_pixels * 100),
            2
        )

        background_percentage = round(
            float(background_pixels / total_pixels * 100),
            2
        )

        mean_intensity = round(
            float(gray_image.mean()),
            2
        )

        std_intensity = round(
            float(gray_image.std()),
            2
        )

        min_intensity = int(gray_image.min())
        max_intensity = int(gray_image.max())

        # Model probability statistics
        mean_probability = round(
            float(probability_map.mean() * 100),
            2
        )

        max_probability = round(
            float(probability_map.max() * 100),
            2
        )

        vessel_probability_values = probability_map[prediction > 0]

        if vessel_probability_values.size > 0:
            mean_vessel_confidence = round(
                float(vessel_probability_values.mean() * 100),
                2
            )

            low_confidence_count = int(
                ((vessel_probability_values >= 0.50) &
                 (vessel_probability_values < 0.75)).sum()
            )

            medium_confidence_count = int(
                ((vessel_probability_values >= 0.75) &
                 (vessel_probability_values < 0.90)).sum()
            )

            high_confidence_count = int(
                (vessel_probability_values >= 0.90).sum()
            )

            vessel_confidence_total = int(
                vessel_probability_values.size
            )

            low_confidence_percentage = round(
                low_confidence_count /
                vessel_confidence_total * 100,
                2
            )

            medium_confidence_percentage = round(
                medium_confidence_count /
                vessel_confidence_total * 100,
                2
            )

            high_confidence_percentage = round(
                high_confidence_count /
                vessel_confidence_total * 100,
                2
            )

        else:
            mean_vessel_confidence = 0.0
            low_confidence_percentage = 0.0
            medium_confidence_percentage = 0.0
            high_confidence_percentage = 0.0

        # Create red vessel overlay
        overlay = image.copy()
        overlay[prediction > 0] = [0, 0, 255]

        blended = cv2.addWeighted(
            image,
            0.7,
            overlay,
            0.3,
            0
        )

        original_base64 = image_to_base64(image)
        result_base64 = image_to_base64(prediction)
        overlay_base64 = image_to_base64(blended)

        return render(
            request,
            "segmentation/home.html",
            {
                "uploaded_name": uploaded_file.name,
                "original_image": original_base64,
                "result_image": result_base64,
                "overlay_image": overlay_base64,
                "vessel_coverage": vessel_coverage,
                "background_percentage": background_percentage,
                "vessel_pixels": vessel_pixels,
                "background_pixels": background_pixels,
                "total_pixels": total_pixels,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
                "min_intensity": min_intensity,
                "max_intensity": max_intensity,
                "mean_probability": mean_probability,
                "max_probability": max_probability,
                "mean_vessel_confidence": mean_vessel_confidence,
                "low_confidence_percentage": low_confidence_percentage,
                "medium_confidence_percentage": medium_confidence_percentage,
                "high_confidence_percentage": high_confidence_percentage,
                "image_width": width,
                "image_height": height,
            }
        )

    return render(
        request,
        "segmentation/home.html"
    )
