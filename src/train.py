import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import numpy as np
import torch
import pandas as pd
from common.metadata_loader import load_metadata, load_fitzpatrick_metadata
from config import DEVICE, CHECKPOINT_FOLDER
from common.data_visualizer import display_basic_analysis
from classification_model.train import train_classification
from segmentation_model.train import train_segmentation

def main():
    print("Starting training process...")
    print(f"Using device: {DEVICE}")

    df_2016, df_2017, df_2018, df_2019, df_2020 = load_metadata()
    df_fitzpatrick = load_fitzpatrick_metadata()
    
    df_2017 = pd.concat([df_2016, df_2017], ignore_index=True)

    df_all = pd.concat([df_2017, df_2018, df_2019, df_2020], ignore_index=True)

    print("\n2016 Dataset Analysis:")
    display_basic_analysis(df_2016)
    print("\n2017 Dataset Analysis:")
    display_basic_analysis(df_2017)
    print("\n2018 Dataset Analysis:")
    display_basic_analysis(df_2018)
    print("\n2019 Dataset Analysis:")
    display_basic_analysis(df_2019)
    print("\n2020 Dataset Analysis:")
    display_basic_analysis(df_2020)
    print("\nFitzpatrick17k Dataset Analysis:")
    display_basic_analysis(df_fitzpatrick)

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)

    train_segmentation(df_all.copy())
    train_classification(df_2017, df_2019, df_2020, df_fitzpatrick)

    print("\nTraining process completed. Checkpoints saved in:", CHECKPOINT_FOLDER)

if __name__ == "__main__":
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

    main()