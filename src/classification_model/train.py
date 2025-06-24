import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import gc

from common.constants import ColumnNames
from common.data_visualizer import display_basic_analysis, display_skin_tone_analysis
from config import *
from classification_model.dataset import ClassificationDataset
from classification_model.transformation import classification_train_transformation, classification_val_transformation
from classification_model.model import EfficientNetClassificationModel
from classification_model.criterion import FocalLoss
from classification_model.evaluate import evaluate_model
from common.metrics import find_best_threshold_balanced
from pipeline.utils.data_sampling import prepare_classification_data

def train_classification(
    df_2017: pd.DataFrame,
    df_2019: pd.DataFrame,
    df_2020: pd.DataFrame,
    df_fitzpatrick: pd.DataFrame,
    use_cache: bool = True
) -> None:
    """
    Train a classification model using preprocessed datasets from multiple sources.
    Includes data preparation, staged unfreezing, threshold tuning, and early stopping.

    Parameters:
    ----------
    df_2017 : pd.DataFrame
        Preprocessed dataset from the 2017 source.
    df_2019 : pd.DataFrame
        Preprocessed dataset from the 2019 source.
    df_2020 : pd.DataFrame
        Preprocessed dataset from the 2020 source.
    df_fitzpatrick : pd.DataFrame
        Preprocessed Fitzpatrick17k dataset, typically containing skin type and image data.
    use_cache : bool, optional (default=True)
        Whether to use cached preprocessed data/models if available.

    Returns:
    -------
    None
        This function performs training and logging internally
    """

    print("\n" + "=" * 50)
    print("Starting classifier training...")
    print("=" * 50 + "\n")
    print("Start preprocessing images...")

    # Prepare training and validation data
    df_train, df_val = prepare_classification_data(df_2017, df_2019, df_2020, df_fitzpatrick, use_cache)

    # Display dataset statistics
    print("\nTrain set analysis:")
    display_basic_analysis(df_train)
    display_skin_tone_analysis(df_train)
    print("\nValidation set analysis:")
    display_basic_analysis(df_val)
    display_skin_tone_analysis(df_val)

    # Create datasets and data loaders
    train_dataset = ClassificationDataset(df_train, transform=classification_train_transformation)
    val_dataset   = ClassificationDataset(df_val, transform=classification_val_transformation)

    train_loader = DataLoader(train_dataset, batch_size=CLASSIFICATION_MODEL_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=CLASSIFICATION_MODEL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize model and training components
    device = torch.device(DEVICE)
    model = EfficientNetClassificationModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1.5e-4,
        total_steps=CLASSIFIER_EPOCHS * len(train_loader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=1e4,
        cycle_momentum=False
    )

    criterion = FocalLoss(alpha=3.4, gamma=2.0)

    torch.cuda.empty_cache()
    gc.collect()

    best_val_loss = float('inf')

    # Freeze all layers except classification head initially
    model.set_requires_grad(model.HEAD_LAYERS)

    # Set up AMP (automatic mixed precision)
    scaler = GradScaler()
    best_thr = 0.50
    patience = 10
    trigger_times = 0

    # ----------------------------
    #         TRAINING LOOP
    # ----------------------------
    print("\nStarting training loop...")
    for epoch in range(CLASSIFIER_EPOCHS):

        # Staged unfreezing of layers
        if epoch == 4:
            # Unfreeze last block and head
            model.set_requires_grad(model.LAST_BLOCK_LAYERS + model.HEAD_LAYERS)
            add_new_params_to_optim(optimizer, model, lr=5e-5)

        elif epoch == 8:
            # Unfreeze all layers and lower learning rate
            model.set_requires_grad(model.ALL_LAYERS)
            for group in optimizer.param_groups:
                group["lr"] = 1e-5
            add_new_params_to_optim(optimizer, model, lr=5e-6)

        # Train for one epoch
        train_loss, train_probs, train_targets = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, scheduler, scaler, device
        )

        # Validate the model
        val_loss, val_targets, val_probs, val_skin = validate_one_epoch(
            epoch, model, val_loader, criterion, device
        )

        # Find best classification threshold based on validation set
        best_thr = find_best_threshold_balanced(val_targets, val_probs, val_skin)

        # Calculate training accuracy
        train_preds = (train_probs > best_thr).astype(int)
        train_acc = 100. * (train_preds == train_targets).sum() / len(train_targets)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

        # Evaluate on validation set with fairness metrics
        evaluate_model(
            all_targets=val_targets,
            all_probabilities=val_probs,
            all_skin_tones=val_skin,
            threshold=best_thr,
            val_loss=val_loss
        )

        # Save model checkpoint
        torch.save({
            ColumnNames.MODEL_STATE_DICT: model.state_dict(),
            ColumnNames.THRESHOLD: best_thr,
        }, CLASSIFICATION_MODEL_CHECKPOINT_PATH)
        print(f"Model checkpoint saved at {CLASSIFICATION_MODEL_CHECKPOINT_PATH}")

        # Early stopping logic
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

        torch.cuda.empty_cache()
        gc.collect()

    print("\nClassification training complete!")

    # Final evaluation on validation set
    evaluate_model(
        all_targets=val_targets,
        all_probabilities=val_probs,
        all_skin_tones=val_skin,
        threshold=best_thr,
        val_loss=val_loss
    )

def add_new_params_to_optim(optim, model, lr):
    """
    Adds newly unfrozen model parameters to an existing optimizer.

    Args:
        optim: torch optimizer
        model: the model with updated requires_grad
        lr: learning rate for newly added parameters
    """
    # Track which parameters are already in the optimizer
    known = {id(p) for group in optim.param_groups for p in group['params']}

    # Find new parameters that require gradients and are not yet in the optimizer
    new_params = [p for p in model.parameters() if p.requires_grad and id(p) not in known]

    # Add new parameters with specified learning rate
    if new_params:
        optim.add_param_group({'params': new_params, 'lr': lr})


def train_one_epoch(epoch, model, train_loader, optimizer, criterion, scheduler, scaler, device):
    """
    Runs a single training epoch with AMP and scheduler.

    Returns:
        Average loss, predicted probabilities, and true targets.
    """
    model.train()
    running_loss = 0.0
    all_logits, all_targets = [], []

    for images, targets, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{CLASSIFIER_EPOCHS}"):
        images = images.float().to(device)
        targets = targets.float().to(device)

        optimizer.zero_grad()

        with autocast(DEVICE):
            logits = model(images).squeeze(-1)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        all_logits.append(torch.sigmoid(logits).detach().cpu())
        all_targets.append(targets.detach().cpu())

    epoch_loss = running_loss / len(train_loader)
    all_probabilities = torch.cat(all_logits).numpy()
    all_targets = torch.cat(all_targets).numpy().astype(int)

    return epoch_loss, all_probabilities, all_targets


def validate_one_epoch(epoch, model, val_loader, criterion, device):
    """
    Runs a validation epoch without gradient tracking.

    Returns:
        Average loss, true labels, predicted probabilities, and skin tone labels.
    """

    model.eval()
    v_loss = 0.0
    all_targets, all_probabilities, all_skin_tones = [], [], []

    with torch.no_grad():
        for images, targets, skin_tones in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{CLASSIFIER_EPOCHS}"):
            images = images.float().to(device)
            targets = targets.float().to(device)
            skin_tones = skin_tones.to(device).long()

            logits = model(images).squeeze(-1)
            v_loss += criterion(logits, targets).item()
            probs = torch.sigmoid(logits)

            all_targets.extend(targets.cpu().numpy().astype(int))
            all_probabilities.extend(probs.cpu().numpy())
            all_skin_tones.extend(skin_tones.cpu().numpy().astype(int))

    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    all_skin_tones = np.array(all_skin_tones)

    return v_loss / len(val_loader), all_targets, all_probabilities, all_skin_tones