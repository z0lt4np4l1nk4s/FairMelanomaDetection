from config import IMAGE_SIZE, PREPROCESSED_FOLDER
from pipeline.preprocessing.classification_preprocessor import ClassificationPreprocessor
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List

# Global preprocessor instance for worker processes
_global_preprocessor = None

def init_worker() -> None:
    """
    Initialize the global preprocessor in each subprocess.
    This avoids reloading models/configurations for every image.
    """
    global _global_preprocessor
    _global_preprocessor = ClassificationPreprocessor(
        output_size=IMAGE_SIZE,
        output_folder=PREPROCESSED_FOLDER,
        use_cache=True
    )

def process_row(row_dict: Dict[str, Any], is_fitzpatrick: bool = False) -> Dict[str, Any]:
    """
    Process a single row (image metadata) using the global preprocessor.

    Parameters:
    ----------
    row_dict : Dict[str, Any]
        A dictionary representing a single row of the DataFrame, typically image metadata.
    is_fitzpatrick : bool, optional
        Whether the image is from the Fitzpatrick17k dataset (may change preprocessing logic).

    Returns:
    -------
    Dict[str, Any]
        Processed image data as a dictionary.
    """
    global _global_preprocessor
    return _global_preprocessor.process_image(row_dict, is_fitzpatrick=is_fitzpatrick)

def parallel_preprocess(
    df: pd.DataFrame,
    is_fitzpatrick: bool = False,
    workers: int = 4
) -> pd.DataFrame:
    """
    Parallel preprocessing of a DataFrame containing image metadata using multiple processes.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame containing metadata for each image to be preprocessed.
    is_fitzpatrick : bool, optional
        Whether the dataset is Fitzpatrick17k, affects processing rules.
    workers : int, optional
        Number of worker processes to use for parallelization.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame with processed image data.
    """
    # Convert DataFrame to a list of dictionaries to avoid pickling issues
    row_dicts: List[Dict[str, Any]] = df.to_dict("records")

    # Use multiprocessing to preprocess data in parallel
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        futures = [executor.submit(process_row, row, is_fitzpatrick) for row in row_dicts]

        processed_results: List[Dict[str, Any]] = []
        for future in tqdm(futures, desc="Processing"):
            processed_results.append(future.result())

    return pd.DataFrame(processed_results)