import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import itertools

from common.constants import ColumnNames
from config import *
from segmentation_model.dataset import SegmentationDataset
from segmentation_model.transformation import segmentation_train_transformation, segmentation_val_transformation
from segmentation_model.model import SegmentationModel
from segmentation_model.criterion import TverskyLoss
from pipeline.preprocessing.segmentation_preprocessor import SegmentationPreprocessor


def train_segmentation(df_all: pd.DataFrame, use_cache: bool = True) -> None:
    """
    Main function to train the segmentation model.

    Args:
        df_all (pd.DataFrame): Complete metadata including image names and paths.
        use_cache (bool): Whether to skip preprocessing if preprocessed data already exists.
    """

    print("\n" + "=" * 50)
    print("Starting segmentation training...")
    print("=" * 50 + "\n")

    # Locate available segmentation mask files
    print("Searching for segmentation masks...")
    available_mask_names = [f.replace("_segmentation.png", "") for f in os.listdir(MASKS_FOLDER) if f.endswith('_segmentation.png')]
    print("Found segmentation masks:", len(available_mask_names))

    # Normalize image names by removing '_downsampled' to match with mask names
    print("Searching for original images...")
    df_all[ColumnNames.IMAGE_NAME] = df_all[ColumnNames.IMAGE_NAME].str.replace('_downsampled', '', regex=False)

    # Drop duplicates, preferring high-quality (non-downsampled) images
    df_unique_images = df_all.sort_values(by=ColumnNames.IMAGE_PATH, key=lambda x: x.str.contains('_downsampled'))
    df_unique_images = df_unique_images.drop_duplicates(ColumnNames.IMAGE_NAME)

    # Keep only records with corresponding masks
    df_masked = df_unique_images[df_unique_images[ColumnNames.IMAGE_NAME].isin(available_mask_names)].copy()

    # Assign mask paths to corresponding images
    df_masked[ColumnNames.MASK_PATH] = df_masked[ColumnNames.IMAGE_NAME].apply(
        lambda name: f"{MASKS_FOLDER}/{name}_segmentation.png"
    )
    print("Found paired images and masks:", len(df_masked))

    # Preprocess images and masks
    print("\nStart preprocessing images...")
    tqdm.pandas(desc="Preprocessing Images")

    # Ensure output folder exists
    if not os.path.exists(PREPROCESSED_MASKS_FOLDER):
        os.makedirs(PREPROCESSED_MASKS_FOLDER)

    # Load from cache if available
    if use_cache and os.path.exists(PREPROCESSED_MASKS_METADATA_PATH):
        df_masked = pd.read_csv(PREPROCESSED_MASKS_METADATA_PATH)

    # Apply preprocessing transformation
    preprocessor = SegmentationPreprocessor(size=SEGMENTATION_IMAGE_SIZE, output_folder=PREPROCESSED_MASKS_FOLDER, use_cache=use_cache)
    df_masked = df_masked.progress_apply(lambda row: preprocessor.process_pair(row), axis=1)

    print("Preprocessing completed.")

    # Save preprocessed metadata
    print("Saving preprocessed metadata...")
    df_masked.to_csv(PREPROCESSED_MASKS_METADATA_PATH, index=False)
    print("Preprocessed metadata saved to:", PREPROCESSED_MASKS_METADATA_PATH)

    # Train-validation split
    df_train, df_val = train_test_split(df_masked, train_size=0.8, random_state=42, shuffle=True)

    # Create dataset objects with corresponding transformations
    train_dataset = SegmentationDataset(dataframe=df_train, transform=segmentation_train_transformation)
    val_dataset = SegmentationDataset(dataframe=df_val, transform=segmentation_val_transformation)

    print("\nTrain samples:", len(train_dataset))
    print("Val samples:  ", len(val_dataset))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=SEGMENTATION_MODEL_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=SEGMENTATION_MODEL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize model and device
    model = SegmentationModel()
    device = torch.device(DEVICE)

    # Freeze all model parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze decoder, head, and later encoder layers for initial training phase
    for param in itertools.chain(
        model.decoder.parameters(),
        model.head.parameters(),
        model.encoder.layer4.parameters(),
        model.encoder.layer3.parameters()
    ):
        param.requires_grad = True

    # Define loss, optimizer, and scheduler
    criterion = TverskyLoss(alpha=0.7, beta=0.3)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    best_val_loss = float('inf')
    early_stop_patience = 5
    epochs_without_improvement = 0
    max_epochs = SEGMENTATION_EPOCHS
    unfreeze_epoch = 3
    fine_tune_lr = 1e-4

    print("\nStarting training loop...")
    for epoch in range(max_epochs):
        train_loss = train_epoch(epoch, model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(epoch, model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{max_epochs}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

        # Unfreeze the entire encoder after warmup epochs
        if epoch + 1 == unfreeze_epoch:
            print("⇢ unfreezing entire encoder and lowering LR → 1e-4")
            for param in model.encoder.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=fine_tune_lr,
                weight_decay=1e-5
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
            )
            print(f"Unfroze encoder.layer3 & layer4; switched to lr=1e-4")

        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SEGMENTATION_MODEL_CHECKPOINT_PATH)
            print(f"Saved model to {SEGMENTATION_MODEL_CHECKPOINT_PATH}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print("Early stopping triggered.")
                break

    print("Segmentation training completed.")


def train_epoch(epoch: int, model, dataloader, optimizer, criterion, device) -> float:
    """
    Runs one training epoch.

    Returns:
        float: Average training loss.
    """

    model.train()

    total_batches = 0
    accumulated_loss = 0.0

    for images, masks in tqdm(dataloader, desc=f"Training {epoch+1}/{SEGMENTATION_EPOCHS}"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        accumulated_loss += loss.item()
        total_batches += 1

    return accumulated_loss / total_batches


@torch.no_grad()
def validate_epoch(epoch: int, model, dataloader, criterion, device) -> float:
    """
    Runs one validation epoch.

    Returns:
        float: Average validation loss.
    """

    model.eval()

    total_batches = 0
    accumulated_loss = 0.0

    for images, masks in tqdm(dataloader, desc=f"Validating {epoch+1}/{SEGMENTATION_EPOCHS}"):
        images, masks = images.to(device), masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        accumulated_loss += loss.item()
        total_batches += 1

    return accumulated_loss / total_batches
