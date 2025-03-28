import os
import base64
import requests
from PIL import Image, ImageDraw

# ===========================
# PUBLIC CONFIDENCE THRESHOLDS
# ===========================
LOW_CONFIDENCE_MIN = 0.50   # Anything below this will be skipped
LOW_CONFIDENCE_MAX = 0.60   # (0.50 - 0.60) => Green
MID_CONFIDENCE_MIN = 0.61
MID_CONFIDENCE_MAX = 0.69   # (0.61 - 0.79) => Orange
HIGH_CONFIDENCE_MIN = 0.70  # (>= 0.80) => Red
# ===========================

# API Endpoint
API_URL = "https://car-damage-1068247786814.us-central1.run.app"

# Input folder path
INPUT_FOLDER = "data/images"

def get_image_path():
    """Fetch the first image found in the input folder."""
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(INPUT_FOLDER, file)
    return None

def encode_image_to_base64(image_path):
    """Convert an image file to a Base64-encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def send_image_to_api(base64_image):
    """Send the Base64 image to the API and return the response."""
    payload = {"image": base64_image}
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        return response.json()  # API should return predictions in JSON
    else:
        return {
            "error": f"API request failed with status {response.status_code}",
            "details": response.text
        }

def draw_bounding_boxes(image_path, predictions, output_path=None):
    """
    Draw bounding boxes from the API response on the image.
    Assumes boxes are in (centerX, centerY, width, height) format.

    We skip any confidence < LOW_CONFIDENCE_MIN,
    then choose colors based on these “public” threshold variables:
      - Green:  LOW_CONFIDENCE_MIN..LOW_CONFIDENCE_MAX
      - Orange: MID_CONFIDENCE_MIN..MID_CONFIDENCE_MAX
      - Red:    >= HIGH_CONFIDENCE_MIN
    """
    # Open the original image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for pred in predictions:
        confidence = pred.get("confidence", 0.0)
        print("CONFIDENCE: ", pred.get("confidence"))
        
        # Skip boxes below LOW_CONFIDENCE_MIN
        if confidence < LOW_CONFIDENCE_MIN:
            continue

        # Determine bounding box color based on confidence
        if LOW_CONFIDENCE_MIN <= confidence <= LOW_CONFIDENCE_MAX:
            box_color = "green"
        elif MID_CONFIDENCE_MIN <= confidence <= MID_CONFIDENCE_MAX:
            box_color = "orange"
        elif confidence >= HIGH_CONFIDENCE_MIN:
            box_color = "red"
        else:
            # If it falls between thresholds (e.g. 0.60 and 0.61),
            # you can decide how to handle it. For safety, skip or pick a default color.
            # Let's skip it for clarity:
            continue

        # Extract center-based coordinates
        x_center = pred["x"]
        y_center = pred["y"]
        box_width = pred["width"]
        box_height = pred["height"]
        # label = pred["class"]  # If you want to display the class name

        # Convert center-based coords to [left, top, right, bottom]
        left = x_center - (box_width / 2)
        top = y_center - (box_height / 2)
        right = x_center + (box_width / 2)
        bottom = y_center + (box_height / 2)

        # Draw bounding box
        draw.rectangle([left, top, right, bottom], outline=box_color, width=2)

        # If you want to draw label + confidence, uncomment and customize:
        from PIL import ImageFont
        font = ImageFont.load_default()
        text = f"{pred['class']} ({confidence:.2f})"
        draw.text((left, top - 12), text, fill=box_color, font=font)

    # Define default output path if not provided
    if not output_path:
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = f"annotated_{name}.png"

    image.save(output_path)
    print(f"✅ Saved annotated image to: {output_path}")
    return output_path

def main():
    """Main function to process the image and send it to the API."""
    image_path = get_image_path()
    if not image_path:
        print("❌ No image found in the input folder.")
        return

    print(f"📷 Using image: {image_path}")
    
    base64_image = encode_image_to_base64(image_path)
    
    print("🚀 Sending image to API...")
    result = send_image_to_api(base64_image)
    
    print("\n🎯 API Response (Prediction):")
    print(result)

    # If there's an error key, stop here
    if "error" in result:
        print("❌ Could not draw bounding boxes due to error in API response.")
        return

    # Draw bounding boxes if predictions exist
    predictions = result.get("predictions", [])
    if predictions:
        draw_bounding_boxes(image_path, predictions)
    else:
        print("❌ No predictions found in the API response.")

if __name__ == "__main__":
    main()
