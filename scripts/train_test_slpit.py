import os
import shutil
import random
import logging

BASE_DIR = './data'
RAW_DIR = os.path.join(BASE_DIR, 'raw')

# Logging configuration
logging.basicConfig(
    filename="./logs/log.log", 
    filemode='a', 
    level=logging.INFO, 
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

def train_test_split(name, split_ratio, sample=150):
    """
    Split the dataset for a specific animal into training, validation, and test sets.
    
    Args:
        name (str): Name of the animal.
        split_ratio (list): List containing train, validation, and test split ratios.
        sample (int): The number of samples to consider for splitting.
    """
    # Set to the raw data path
    DATA_PATH = os.path.join(RAW_DIR, name)
    files = [filename[:-4] for filename in os.listdir(DATA_PATH) if filename.endswith(('.txt', '.TXT'))]

    # Shuffle the data
    random.seed(42)
    random.shuffle(files)

    # Adjust sample size if larger than available data
    sample = min(sample, len(files))

    # Randomly take sample images
    files = random.sample(files, sample)

    # Calculate the split sizes based on the split ratio
    train_size = int(sample * split_ratio[0])
    val_size = int(sample * split_ratio[1])
    
    # Split into train, validation, and test sets
    train_sets = files[:train_size]
    val_sets = files[train_size:train_size + val_size]
    test_sets = files[train_size + val_size:]

    # Split and save the dataset into images and labels
    images_labels_split(DATA_PATH, train_sets, name, mode="train")  # For training set
    images_labels_split(DATA_PATH, val_sets, name, mode="val")  # For validation set
    images_labels_split(DATA_PATH, test_sets, name, mode="test")  # For test set

    logging.info(f"Successfully split {name} dataset into train, validation, and test sets.")

def images_labels_split(data_path, subfile, name, mode):
    """
    Split and copy the images and labels into respective train/val/test directories.
    
    Args:
        data_path (str): Path to the raw files.
        subfile (list): List of randomly selected files.
        name (str): Name of the animal.
        mode (str): Type of sets (train, val, test).
    """
    # Select file names from all files
    raw_files = os.listdir(data_path)
    data = [file for file in raw_files if file[:-4] in subfile]

    # Filter images and labels
    images = [img for img in data if img.lower().endswith(('.jpg', '.jpeg'))]
    labels = [label for label in data if label.lower().endswith('.txt')]

    # Ensure images and labels match in count
    if len(images) != len(labels):
        logging.warning(f"Mismatch in image and label count for {name} in {mode} set.")
        return

    # Copy image files to the respective directories
    img_dir = os.path.join(BASE_DIR, f'images/{mode}/{name}')
    os.makedirs(img_dir, exist_ok=True)
    
    try:
        for img in images:
            raw_img_path = os.path.join(data_path, img)
            shutil.copy(raw_img_path, img_dir)
        logging.info(f"Copied {len(images)} images to {mode} set for {name}.")
    except Exception as e:
        logging.warning(f"Error copying images for {name} in {mode} set: {e}")

    # Copy label files to the respective directories
    label_dir = os.path.join(BASE_DIR, f'labels/{mode}/{name}')
    os.makedirs(label_dir, exist_ok=True)
    
    try:
        for label in labels:
            raw_label_path = os.path.join(data_path, label)
            shutil.copy(raw_label_path, label_dir)
        logging.info(f"Copied {len(labels)} labels to {mode} set for {name}.")
    except Exception as e:
        logging.warning(f"Error copying labels for {name} in {mode} set: {e}")

if __name__ == "__main__":
    # Set split ratio (70% train, 15% validation, 15% test)
    split_ratio = [0.7, 0.15]

    # List all animals from the raw directory
    animals = os.listdir(RAW_DIR)

    # Process each animal's dataset
    for animal in animals:
        train_test_split(animal, split_ratio)

    logging.info("Successfully completed the dataset splitting for all animals.")
