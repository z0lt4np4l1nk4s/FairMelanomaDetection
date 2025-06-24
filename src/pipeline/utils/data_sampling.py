import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from common.constants import ColumnNames, Constants
from typing import Tuple
from config import PREPROCESSED_METADATA_2017_PATH, PREPROCESSED_METADATA_2019_PATH, PREPROCESSED_METADATA_2020_PATH, PREPROCESSED_METADATA_FITZPATRICK_PATH, NUM_WORKERS
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from pipeline.utils.preprocessing_parallel import parallel_preprocess

def balance_by_skin_tone(
    df: pd.DataFrame,
    sample_size: int
) -> pd.DataFrame:
    """
    Return a DataFrame balanced by skin tone.

    This function preserves all samples from underrepresented skin tones,
    and downsamples the rest to ensure the total number of samples matches `sample_size`.
    
    Args:
        df (pd.DataFrame): The full dataset, containing a skin tone column.
        sample_size (int): Total number of samples to return.

    Returns:
        pd.DataFrame: A DataFrame balanced by skin tone.
    """
    # Select all samples belonging to underrepresented skin tones
    underrepresented_samples = df[df[ColumnNames.SKIN_TONE].isin(Constants.UNDERREPRESENTED_SKIN_TONES)]

    # Select the remaining (overrepresented) samples
    overrepresented_samples = df[~df[ColumnNames.SKIN_TONE].isin(Constants.UNDERREPRESENTED_SKIN_TONES)]

    # Calculate the number of overrepresented samples needed
    max_overrepresented = sample_size - len(underrepresented_samples)

    # Randomly sample the overrepresented set if it's too large
    if len(overrepresented_samples) > max_overrepresented:
        overrepresented_samples = overrepresented_samples.sample(n=max_overrepresented, random_state=42)

    # Combine both sets into a balanced DataFrame
    return pd.concat([underrepresented_samples, overrepresented_samples], ignore_index=True)

def build_limited_dataset(
    df: pd.DataFrame,
    benign_sample_size: int = 20000,
    max_images_per_patient: int = 20,
    val_split_ratio: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct a balanced dataset by:
    - Including all images from patients with malignant cases or underrepresented skin tones
    - Sampling a fixed number of benign images from the remaining patients
    - Limiting the number of images per patient
    - Splitting the dataset into training and validation sets, grouped by patient
    
    Args:
        df (pd.DataFrame): The full dataset with patient, skin tone, and target labels.
        benign_sample_size (int): Max number of benign images to sample from non-priority patients.
        max_images_per_patient (int): Maximum number of images to keep per patient.
        val_split_ratio (float): Fraction of the dataset to use for validation.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing train and validation DataFrames.
    """
    patient_id_col = ColumnNames.PATIENT_ID
    target_col = ColumnNames.TARGET
    skin_tone_col = ColumnNames.SKIN_TONE

    # Identify patients with malignant cases
    malignant_patient_ids = set(df.loc[df[target_col] == 1, patient_id_col].unique())

    # Identify patients with underrepresented skin tones
    underrepresented_patient_ids = set(
        df.loc[df[skin_tone_col].isin(Constants.UNDERREPRESENTED_SKIN_TONES), patient_id_col].unique()
    )

    # Combine the above patients as priority patients
    priority_patient_ids = malignant_patient_ids.union(underrepresented_patient_ids)

    # Keep all images for these priority patients
    priority_samples = df[df[patient_id_col].isin(priority_patient_ids)]

    # Get benign samples from remaining patients
    benign_pool = df[
        (df[target_col] == 0) & (~df[patient_id_col].isin(priority_patient_ids))
    ]
    benign_samples = benign_pool.sample(
        n=min(benign_sample_size, len(benign_pool)),
        random_state=random_state
    )

    # Combine priority and benign samples
    combined_df = pd.concat([priority_samples, benign_samples], ignore_index=True)

    # Limit the number of images per patient
    def limit_images(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) <= max_images_per_patient:
            return group
        return group.sample(n=max_images_per_patient, random_state=random_state)

    limited_df = (
        combined_df
        .groupby(patient_id_col, group_keys=False)
        .apply(limit_images)
        .reset_index(drop=True)
    )

    # Split into train and validation sets using patient ID as group
    group_splitter = GroupShuffleSplit(n_splits=1, test_size=val_split_ratio, random_state=random_state)
    train_idx, val_idx = next(group_splitter.split(limited_df, groups=limited_df[patient_id_col]))

    train_df = limited_df.iloc[train_idx].reset_index(drop=True)
    val_df = limited_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df

def prepare_classification_data(
    df_2017: pd.DataFrame,
    df_2019: pd.DataFrame,
    df_2020: pd.DataFrame,
    df_fitzpatrick: pd.DataFrame,
    use_cache: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares and returns training and validation datasets for classification.
    Applies preprocessing, stratified sampling, balancing, and upsampling.

    Args:
        df_2017 (pd.DataFrame): Metadata for the 2017 dataset.
        df_2019 (pd.DataFrame): Metadata for the 2019 dataset.
        df_2020 (pd.DataFrame): Metadata for the 2020 dataset.
        df_fitzpatrick (pd.DataFrame): Metadata for the Fitzpatrick skin tone dataset.
        use_cache (bool): Whether to load from cached CSV files if they exist.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the final training and validation datasets.
    """

    tqdm.pandas(desc="Preprocessing Images")

    # Load cached metadata if available and caching is enabled
    if use_cache:
        if os.path.exists(PREPROCESSED_METADATA_2017_PATH):
            df_2017 = pd.read_csv(PREPROCESSED_METADATA_2017_PATH)
        if os.path.exists(PREPROCESSED_METADATA_2019_PATH):
            df_2019 = pd.read_csv(PREPROCESSED_METADATA_2019_PATH)
        if os.path.exists(PREPROCESSED_METADATA_2020_PATH):
            df_2020 = pd.read_csv(PREPROCESSED_METADATA_2020_PATH)
        if os.path.exists(PREPROCESSED_METADATA_FITZPATRICK_PATH):
            df_fitzpatrick = pd.read_csv(PREPROCESSED_METADATA_FITZPATRICK_PATH)

    # Apply preprocessing to each dataset
    df_2017 = parallel_preprocess(df_2017, is_fitzpatrick=False, workers=NUM_WORKERS)
    df_2019 = parallel_preprocess(df_2019, is_fitzpatrick=False, workers=NUM_WORKERS)
    df_2020 = parallel_preprocess(df_2020, is_fitzpatrick=False, workers=NUM_WORKERS)
    df_fitzpatrick = parallel_preprocess(df_fitzpatrick, is_fitzpatrick=True, workers=NUM_WORKERS)

    print("Preprocessing completed.")
    print("Saving preprocessed metadata...")

    # Save updated metadata to CSV
    df_2017.to_csv(PREPROCESSED_METADATA_2017_PATH, index=False)
    df_2019.to_csv(PREPROCESSED_METADATA_2019_PATH, index=False)
    df_2020.to_csv(PREPROCESSED_METADATA_2020_PATH, index=False)
    df_fitzpatrick.to_csv(PREPROCESSED_METADATA_FITZPATRICK_PATH, index=False)

    # Separate benign and malignant cases for sampling
    df_benign_2017 = df_2017[df_2017[ColumnNames.TARGET] == 0]
    df_benign_2019 = df_2019[df_2019[ColumnNames.TARGET] == 0]
    df_benign_2020 = df_2020[df_2020[ColumnNames.TARGET] == 0]

    df_malignant_2017 = df_2017[df_2017[ColumnNames.TARGET] == 1]
    df_malignant_2019 = df_2019[df_2019[ColumnNames.TARGET] == 1]
    df_malignant_2020 = df_2020[df_2020[ColumnNames.TARGET] == 1]

    # Combine all benign and malignant data for reference or optional use
    df_benign_all = pd.concat([df_benign_2017, df_benign_2019, df_benign_2020], ignore_index=True)
    df_malignant_all = pd.concat([df_malignant_2017, df_malignant_2019, df_malignant_2020], ignore_index=True)
    df_final = pd.concat([df_benign_all, df_malignant_all], ignore_index=True)

    print("Total count:", len(df_final))

    # --- Sample training/validation sets ---

    # Sample from 2020 dataset with balanced benign examples and patient limit
    df_2020_train, df_2020_val = build_limited_dataset(
        df_2020,
        benign_sample_size=20000,
        max_images_per_patient=20,
        val_split_ratio=0.2,
        random_state=42
    )

    # Balance benign samples in 2019 set by skin tone
    df_2019_benign_sampled = balance_by_skin_tone(df_benign_2019, 10000)
    df_2019_limited = pd.concat([df_2019_benign_sampled, df_malignant_2019], ignore_index=True)

    # Stratified split for 2019 set based on GROUP column
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in sss.split(df_2019_limited, df_2019_limited[ColumnNames.GROUP]):
        df_2019_train = df_2019_limited.iloc[train_idx].reset_index(drop=True)
        df_2019_val = df_2019_limited.iloc[val_idx].reset_index(drop=True)

    # Stratified split for 2017 set
    for train_idx, val_idx in sss.split(df_2017, df_2017[ColumnNames.GROUP]):
        df_2017_train = df_2017.iloc[train_idx]
        df_2017_val = df_2017.iloc[val_idx]

    # Stratified split for Fitzpatrick dataset
    df_fitzpatrick_train, df_fitzpatrick_val = train_test_split(
        df_fitzpatrick,
        test_size=0.4,
        random_state=42,
        stratify=df_fitzpatrick[ColumnNames.TARGET]
    )

    # Combine all training and validation datasets
    df_train = pd.concat([df_2017_train, df_2019_train, df_2020_train, df_fitzpatrick_train], ignore_index=True)
    df_val = pd.concat([df_2017_val, df_2019_val, df_2020_val, df_fitzpatrick_val], ignore_index=True)

    # --- Apply custom upsampling for fairness & class balance ---

    # Filter subsets for targeted upsampling
    df_2017_train_malignant = df_2017_train[
        (df_2017_train[ColumnNames.TARGET] == 1) &
        (~df_2017_train[ColumnNames.SKIN_TONE].isin(Constants.UNDERREPRESENTED_SKIN_TONES))
    ]
    df_2019_train_malignant = df_2019_train[
        (df_2019_train[ColumnNames.TARGET] == 1) &
        (~df_2019_train[ColumnNames.SKIN_TONE].isin(Constants.UNDERREPRESENTED_SKIN_TONES))
    ]
    df_2020_train_malignant = df_2020_train[
        (df_2020_train[ColumnNames.TARGET] == 1) &
        (~df_2020_train[ColumnNames.SKIN_TONE].isin(Constants.UNDERREPRESENTED_SKIN_TONES))
    ]
    df_train_medium_malignant = df_train[
        (df_train[ColumnNames.TARGET] == 1) &
        (df_train[ColumnNames.SKIN_TONE] == Constants.MEDIUM_SKIN_TONE)
    ]
    df_train_dark = df_train[df_train[ColumnNames.SKIN_TONE] == Constants.DARK_SKIN_TONE]
    df_fitzpatrick_benign = df_fitzpatrick_train[df_fitzpatrick_train[ColumnNames.TARGET] == 0]

    # Upsample selected subsets
    df_2017_train_malignant_up = pd.concat([df_2017_train_malignant] * 3, ignore_index=True)
    df_2019_train_malignant_up = pd.concat([df_2019_train_malignant] * 1, ignore_index=True)
    df_2020_train_malignant_up = pd.concat([df_2020_train_malignant] * 3, ignore_index=True)
    df_train_medium_malignant_up = pd.concat([df_train_medium_malignant] * 2, ignore_index=True)
    df_train_dark_up = pd.concat([df_train_dark] * 3, ignore_index=True)
    df_fitzpatrick_benign_up = pd.concat([df_fitzpatrick_benign] * 2, ignore_index=True)

    # Combine upsampled data with original training set
    df_train = pd.concat([
        df_train,
        df_2017_train_malignant_up,
        df_2019_train_malignant_up,
        df_2020_train_malignant_up,
        df_train_medium_malignant_up,
        df_train_dark_up,
        df_fitzpatrick_benign_up
    ], ignore_index=True)

    # Shuffle the final training set
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_train, df_val
