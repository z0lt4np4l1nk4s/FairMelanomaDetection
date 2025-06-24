class Constants:
    """
    General constants used across the project, including skin tone categories,
    diagnostic labels, and device options.
    """

    # Skin tone classification constants
    UNKNOWN_SKIN_TONE = -1
    VERY_LIGHT_SKIN_TONE = 0
    LIGHT_SKIN_TONE = 1
    MEDIUM_LIGHT_SKIN_TONE = 2
    MEDIUM_SKIN_TONE = 3
    DARK_SKIN_TONE = 4

    # Diagnostic labels
    MALIGNANT = "malignant"
    BENIGN = "benign"

    # Skin tone groupings
    DARK_SKIN_TONES = [DARK_SKIN_TONE]
    LIGHT_SKIN_TONES = [VERY_LIGHT_SKIN_TONE, LIGHT_SKIN_TONE]
    UNDERREPRESENTED_SKIN_TONES = [DARK_SKIN_TONE, MEDIUM_SKIN_TONE]
    ALL_SKIN_TONES = [
        VERY_LIGHT_SKIN_TONE,
        LIGHT_SKIN_TONE,
        MEDIUM_LIGHT_SKIN_TONE,
        MEDIUM_SKIN_TONE,
        DARK_SKIN_TONE
    ]

    # Device options
    CUDA = 'cuda'
    CPU = 'cpu'


class ColumnNames:
    """
    Standardized column names for working with Pandas DataFrames.
    """
    
    IMAGE_NAME = "image_name"
    IMAGE_PATH = "image_path"
    MASK_PATH = "mask_path"
    ORIGINAL_IMAGE_PATH = "original_image_path"
    ORIGINAL_MASK_PATH = "original_mask_path"
    SKIN_TONE = "skin_tone"
    TARGET = "target"
    YEAR = "year"
    PATIENT_ID = "patient_id"
    GROUP = "group"
    AGE_APPROX = 'age_approx'
    SEX = 'sex'
    ANATOM_SITE_GENERAL_CHALLENGE = 'anatom_site_general_challenge'
    BASE_NAME = 'base_name'
    THRESHOLD = 'threshold'
    MODEL_STATE_DICT = 'model_state_dict'
