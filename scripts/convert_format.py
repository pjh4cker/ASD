import os
import glob
import logging
from PIL import Image

# Directory where raw data (images and labels) are stored
RAW_DIR = './data/raw'

# Logging configuration
logging.basicConfig(
    filename="./logs/log.log", 
    filemode='w', 
    level=logging.INFO, 
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

def pascal_to_yolo(label_path):
    """
    Convert Pascal VOC labels to YOLO format and save back to the same label file.

    Args:
        label_path (str): Path to the Pascal VOC format label file (.txt).
    """
    # Retrieve corresponding image path (jpg or JPG)
    img_path = label_path.replace(".txt", ".jpg")
    if not os.path.exists(img_path):
        img_path = label_path.replace(".txt", ".JPG")
    
    # Read the image to get dimensions
    try:
        img = Image.open(img_path)
        width, height = img.size
    except Exception as e:
        logging.warning(f"Image not found for label: {label_path}. Error: {e}")
        return

    # Read the label file (Pascal format)
    with open(label_path, 'r') as f:
        labels = f.readlines()

    bbox_coordinates = []

    for label in labels:
        label = label.strip().split(" ")
        # Convert the label values to float if possible
        label = [float(item) if item.replace(".", "", 1).isdigit() else item for item in label]

        # Check if the bounding box dimensions are normalized
        if all(is_normalized(dim) for dim in label[1:]):
            logging.info(f"Label {label_path} already normalized, skipping conversion.")
            return
        else:
            # Convert bounding box to YOLO format
            yolo_format = convert_bbox_to_yolo(label, width, height)
            bbox_coordinates.append(yolo_format)

    # Write YOLO format labels back to the same file
    with open(label_path, 'w') as f:
        for label in bbox_coordinates:
            f.write(f"{label}\n")

    logging.info(f"Converted {label_path} to YOLO format.")


def is_normalized(bound_box_dim):
    """
    Check if a bounding box dimension is normalized between 0 and 1.

    Args:
        bound_box_dim (float): Bounding box dimension value.
    
    Returns:
        bool: True if normalized, False otherwise.
    """
    return 0.0 <= bound_box_dim <= 1.0


def convert_bbox_to_yolo(label, img_width, img_height):
    """
    Convert Pascal VOC bounding box format to YOLO format.

    Args:
        label (list): Pascal VOC label with class and bounding box values.
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        str: YOLO format label string.
    """
    # Class labels mapping
    class_labels = {
        "buffalo": 0,
        "elephant": 1,
        "rhino": 2,
        "zebra": 3,
        "cheetah": 4,
        "fox": 5,
        "jaguar": 6,
        "tiger": 7,
        "lion": 8,
        "panda": 9
    }

    # Extract values from Pascal format
    class_label = class_labels.get(label[0].lower(), -1)
    if class_label == -1:
        logging.warning(f"Unknown class label {label[0]} in {label}")
        return ""

    x_min, y_min, x_max, y_max = label[1:5]

    # Calculate YOLO format (center coordinates and normalized width/height)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # Normalize the coordinates with respect to the image dimensions
    norm_center_x = center_x / img_width
    norm_center_y = center_y / img_height
    norm_width = width / img_width
    norm_height = height / img_height

    # Return formatted YOLO string
    yolo_format = f"{class_label} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
    return yolo_format


if __name__ == "__main__":
    # List of animal directories inside the raw data folder
    animals = os.listdir(RAW_DIR)

    for animal in animals:
        # Get all label files for the current animal
        path_labels = glob.glob(f'{RAW_DIR}/{animal.lower()}/*.txt')
        for path_label in path_labels:
            # Convert each Pascal VOC label to YOLO format
            pascal_to_yolo(path_label)

    logging.info("Successfully completed conversion from Pascal VOC to YOLO format.")
