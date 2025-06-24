import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from classification_model.transformation import classification_val_transformation
from classification_model.model import EfficientNetClassificationModel
from classification_model.dataset import ClassificationTestDataset
from config import CLASSIFICATION_MODEL_CHECKPOINT_PATH, CLASSIFICATION_MODEL_BATCH_SIZE, DEVICE, NUM_WORKERS
from common.constants import ColumnNames


def load_model(checkpoint_path: str, device: torch.device):
    """
    Loads the pretrained classification model and its best threshold from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        device (torch.device): Torch device to map the model to.

    Returns:
        Tuple: (model in eval mode, threshold value for prediction)
    """

    # Load checkpoint dictionary from file
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Initialize model and load state dict
    model = EfficientNetClassificationModel().to(device)
    model.load_state_dict(checkpoint[ColumnNames.MODEL_STATE_DICT])
    model.eval()

    # Load threshold, default to 0.5 if not available
    threshold = checkpoint.get(ColumnNames.THRESHOLD, 0.5)

    return model, threshold


def get_image_paths(input_folder: str):
    """
    Retrieves and sorts image file paths from the specified folder.

    Args:
        input_folder (str): Path to folder containing images.

    Returns:
        List[str]: Sorted list of valid image file paths.
    """

    valid_extensions = ('.jpg', '.jpeg')
    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
                   if f.lower().endswith(valid_extensions)]
    return sorted(image_files)


def main(args) -> None:
    """
    Main function to run prediction on input images and save results to CSV.
    """

    # Set computation device
    device = torch.device(DEVICE)

    # Retrieve all image paths from the input folder
    image_paths = get_image_paths(args.input_folder)
    if not image_paths:
        raise ValueError("No valid images found in the specified folder.")

    # Create dataset and data loader
    dataset = ClassificationTestDataset(image_paths, transform=classification_val_transformation)
    dataloader = DataLoader(dataset, batch_size=CLASSIFICATION_MODEL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Load trained model and optimal threshold
    model, threshold = load_model(CLASSIFICATION_MODEL_CHECKPOINT_PATH, device)

    # List to store prediction dictionaries
    predictions = []

    # Inference loop
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Predicting"):
            images = images.to(device).float()
            outputs = model(images).squeeze(1)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            
            # Apply threshold to get binary predictions
            binary_predictions = (probabilities > threshold).astype(int)

            # Store image name and corresponding binary prediction
            for file_name, prediction in zip(filenames, binary_predictions):
                name_without_extension = os.path.splitext(file_name)[0]
                predictions.append({
                    ColumnNames.IMAGE_NAME: name_without_extension,
                    ColumnNames.TARGET: int(prediction)
                })
    
    # Convert prediction list to DataFrame and save as CSV
    df_pred = pd.DataFrame(predictions)
    df_pred.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set up argument parser for CLI usage
    parser = argparse.ArgumentParser(description="Script for predicting melanoma lesions from images.")
    parser.add_argument("input_folder", type=str, help="Folder containing images for prediction.")
    parser.add_argument("output_csv", type=str, help="Output CSV file to save predictions. Columns: image_name, target")
    args = parser.parse_args()
    main(args)
