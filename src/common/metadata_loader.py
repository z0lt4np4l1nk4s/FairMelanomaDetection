import os
import re
import pandas as pd
from collections import defaultdict
from typing import List, Tuple
from common.constants import ColumnNames, Constants
from config import DUPLICATE_IMAGES_FILE_PATH, TRAIN_FITZPATRICK_IMAGE_FOLDER, METADATA_2016_PATH, METADATA_2017_PATH, METADATA_2018_PATH, METADATA_2019_PATH, METADATA_2020_PATH, METADATA_FITZPATRICK_PATH
from config import TRAIN_2016_IMAGE_FOLDER, TRAIN_2017_IMAGE_FOLDER, TRAIN_2018_IMAGE_FOLDER, TRAIN_2019_IMAGE_FOLDER, TRAIN_2020_IMAGE_FOLDER
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO

def load_metadata() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads metadata from CSV files, removes duplicates, assigns consistent columns,
    and prepares the final metadata DataFrames for each year.

    Returns:
        Tuple of DataFrames for years 2016, 2017, 2018, 2019, and 2020.
    """
    
    print("\nLoading metadata...")

    # Initialize storage for duplicate images found per year
    duplicate_images_per_year = defaultdict(list)

    # Regex patterns for matching filenames based on year
    duplicate_patterns = {
        "2016": r"ISBI2016_ISIC_Part3_Training_Data/(.+?)\.jpg",
        "2017": r"ISIC-2017_Training_Data/(.+?)\.jpg",
        "2018": r"ISIC2018_Task3_Training_Input/(.+?)\.jpg",
        "2019": r"ISIC_2019_Training_Input/(.+?)\.jpg",
        "2020": r"ISIC_2020_Training_JPEG/train/(.+?)\.jpg"
    }

    # Parse duplicates file
    with open(DUPLICATE_IMAGES_FILE_PATH, 'r') as file:
        for line in file:
            for year, pattern in duplicate_patterns.items():
                match = re.search(pattern, line)
                if match:
                    duplicate_images_per_year[year].append(match.group(1))

    # Print duplicate summary
    print("\nDuplicate images per year:")
    for year, images in duplicate_images_per_year.items():
        print(f"{year}: {len(images)} images")
    print(f"Total duplicates: {sum(len(imgs) for imgs in duplicate_images_per_year.values())}")

    # Load metadata CSVs
    metadata_by_year = {
        "2016": pd.read_csv(METADATA_2016_PATH, header=None),
        "2017": pd.read_csv(METADATA_2017_PATH),
        "2018": pd.read_csv(METADATA_2018_PATH),
        "2019": pd.read_csv(METADATA_2019_PATH),
        "2020": pd.read_csv(METADATA_2020_PATH)
    }

    # Mapping of column names for different years
    image_id_columns = {
        "2016": 0,
        "2017": "image_id",
        "2018": "image",
        "2019": "image",
        "2020": "image_name"
    }

    # Remove duplicate images from metadata
    print("\nCleaning metadata:")
    for year, dataframe in metadata_by_year.items():
        if year in duplicate_images_per_year:
            column_name = image_id_columns[year]
            cleaned_df = _delete_rows_with_ids(dataframe, duplicate_images_per_year[year], column_name)
            print(f"{year}: {len(dataframe) - len(cleaned_df)} rows deleted, {len(cleaned_df)} remaining")
            metadata_by_year[year] = cleaned_df

    # Standardize and map metadata for each year
    common_columns = [ColumnNames.IMAGE_NAME, ColumnNames.TARGET, ColumnNames.YEAR]

    df_2016 = metadata_by_year["2016"].copy()
    df_2016.columns = [ColumnNames.IMAGE_NAME, ColumnNames.TARGET]
    df_2016[ColumnNames.TARGET] = df_2016[ColumnNames.TARGET].map({
        Constants.MALIGNANT: 1, Constants.BENIGN: 0
    }).astype(int)
    df_2016[ColumnNames.YEAR] = 2016
    df_2016 = df_2016[common_columns]

    df_2017 = metadata_by_year["2017"].copy()
    df_2017 = df_2017.rename(columns={"image_id": ColumnNames.IMAGE_NAME})
    df_2017[ColumnNames.TARGET] = df_2017["melanoma"].astype(int)
    df_2017[ColumnNames.YEAR] = 2017
    df_2017 = df_2017[common_columns]

    df_2018 = metadata_by_year["2018"].copy()
    df_2018 = df_2018.rename(columns={"image": ColumnNames.IMAGE_NAME})
    df_2018[ColumnNames.TARGET] = df_2018[["MEL", "BCC", "AKIEC"]].sum(axis=1).clip(upper=1).astype(int)
    df_2018[ColumnNames.YEAR] = 2018
    df_2018 = df_2018[common_columns]

    df_2019 = metadata_by_year["2019"].copy()
    df_2019 = df_2019.rename(columns={"image": ColumnNames.IMAGE_NAME})
    df_2019[ColumnNames.TARGET] = df_2019[["MEL", "BCC", "AK", "SCC"]].sum(axis=1).clip(upper=1).astype(int)
    df_2019[ColumnNames.YEAR] = 2019
    df_2019 = df_2019[common_columns]

    df_2020 = metadata_by_year["2020"].copy()
    df_2020[ColumnNames.TARGET] = df_2020[ColumnNames.TARGET].astype(int)
    df_2020[ColumnNames.YEAR] = 2020
    df_2020 = df_2020[common_columns]

    # Combine all years
    combined_metadata = pd.concat([df_2016, df_2017, df_2018, df_2019, df_2020], ignore_index=True)
    combined_metadata = combined_metadata.drop_duplicates(subset=[ColumnNames.IMAGE_NAME])
    combined_metadata = combined_metadata.reset_index(drop=True)

    # Handle downsampled duplicates
    combined_metadata = _filter_downsampled_duplicates(combined_metadata)

    # Add full image paths
    path_columns = [ColumnNames.IMAGE_NAME, ColumnNames.TARGET, ColumnNames.YEAR, ColumnNames.IMAGE_PATH]

    def build_image_path(row: pd.Series, base_folder: str) -> str:
        return f"{base_folder}/{row[ColumnNames.IMAGE_NAME]}.jpg"

    df_2016 = combined_metadata[combined_metadata[ColumnNames.YEAR] == 2016].copy()
    df_2016[ColumnNames.IMAGE_PATH] = df_2016.apply(lambda row: build_image_path(row, TRAIN_2016_IMAGE_FOLDER), axis=1)
    df_2016 = df_2016[path_columns]

    df_2017 = combined_metadata[combined_metadata[ColumnNames.YEAR] == 2017].copy()
    df_2017[ColumnNames.IMAGE_PATH] = df_2017.apply(lambda row: build_image_path(row, TRAIN_2017_IMAGE_FOLDER), axis=1)
    df_2017 = df_2017[path_columns]

    df_2018 = combined_metadata[combined_metadata[ColumnNames.YEAR] == 2018].copy()
    df_2018[ColumnNames.IMAGE_PATH] = df_2018.apply(lambda row: build_image_path(row, TRAIN_2018_IMAGE_FOLDER), axis=1)
    df_2018 = df_2018[path_columns]

    df_2019 = combined_metadata[combined_metadata[ColumnNames.YEAR] == 2019].copy()
    df_2019[ColumnNames.IMAGE_PATH] = df_2019.apply(lambda row: build_image_path(row, TRAIN_2019_IMAGE_FOLDER), axis=1)
    df_2019 = df_2019[path_columns]

    df_2020_final = combined_metadata[combined_metadata[ColumnNames.YEAR] == 2020].copy()
    df_2020_temp = metadata_by_year["2020"]
    df_2020_final = df_2020_final.merge(df_2020_temp[[ColumnNames.IMAGE_NAME, ColumnNames.PATIENT_ID]],
                                        on=ColumnNames.IMAGE_NAME, how="left")
    df_2020_final[ColumnNames.IMAGE_PATH] = df_2020_final.apply(lambda row: build_image_path(row, TRAIN_2020_IMAGE_FOLDER), axis=1)
    df_2020_final = df_2020_final[path_columns + [ColumnNames.PATIENT_ID]]

    return df_2016, df_2017, df_2018, df_2019, df_2020_final

def load_fitzpatrick_metadata():
    print("\nLoading Fitzpatrick metadata...")
    df = pd.read_csv(METADATA_FITZPATRICK_PATH)

    # filter the DataFrame to include only rows with hashes in df_skin_tones['image_name']
    df = df[df['md5hash'].isin(FITZPATRICK_IMAGE_HASHES)].copy()

    # check if images are already downloaded, if not download them
    if not os.path.exists(TRAIN_FITZPATRICK_IMAGE_FOLDER) or not os.listdir(TRAIN_FITZPATRICK_IMAGE_FOLDER):
        print("Downloading images...")
        os.makedirs(TRAIN_FITZPATRICK_IMAGE_FOLDER, exist_ok=True)
        
        success_count = 0
        failure_count = 0
        
        # Wrap the loop with tqdm to display a progress bar
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Downloading Images", unit="image"):
            image_url = row['url']
            save_path = os.path.join(TRAIN_FITZPATRICK_IMAGE_FOLDER, f"{row['md5hash']}.jpg")
            try:
                _download_image(image_url, save_path)
                success_count += 1
            except Exception:
                failure_count += 1
        
        print(f"Download complete. Successful: {success_count}, Failed: {failure_count}")



    malignant_labels = [
        'melanoma',
        'superficial spreading melanoma ssm',
        'lentigo maligna',
        'malignant melanoma',
        'basal cell carcinoma',
        'squamous cell carcinoma',
        'actinic keratosis'
    ]

    benign_labels = [
        'nevocytic nevus', 'congenital nevus',
        'seborrheic keratosis',
        'dermatofibroma',
        'port wine stain'
    ]

    # Create the target column based on labels
    df[ColumnNames.TARGET] = -1  # Default value for labels not in either list
    df.loc[df['label'].isin(benign_labels), ColumnNames.TARGET] = 0
    df.loc[df['label'].isin(malignant_labels), ColumnNames.TARGET] = 1
    
    # Only keep rows where target is 0 or 1
    df = df[df[ColumnNames.TARGET].isin([0, 1])]
    
    # Create the new DataFrame with required columns
    df = pd.DataFrame({
        ColumnNames.IMAGE_NAME: df['md5hash'],
        ColumnNames.TARGET: df[ColumnNames.TARGET],
        ColumnNames.YEAR: -1,
        ColumnNames.IMAGE_PATH: df['md5hash'].apply(lambda x: f"{TRAIN_FITZPATRICK_IMAGE_FOLDER}/{x}.jpg"),
    })

    return df

def _delete_rows_with_ids(df: pd.DataFrame, id_list: List[str], column_name: str = ColumnNames.IMAGE_NAME) -> pd.DataFrame:
    """
    Deletes rows from a DataFrame where the specified column matches any ID from a list.
    """
    mask = ~df[column_name].isin(id_list)
    return df[mask]

def _filter_downsampled_duplicates(df: pd.DataFrame, image_col: str = ColumnNames.IMAGE_NAME) -> pd.DataFrame:
    """
    Removes '_downsampled' images if their original version exists.
    """
    df_copy = df.copy()
    df_copy[ColumnNames.BASE_NAME] = df_copy[image_col].apply(
        lambda name: name.replace("_downsampled", "") if name.endswith("_downsampled") else name
    )

    indices_to_keep = []
    for base, group in df_copy.groupby(ColumnNames.BASE_NAME):
        if (group[image_col] == base).any():
            indices_to_keep.extend(group[group[image_col] == base].index.tolist())
        else:
            indices_to_keep.extend(group.index.tolist())

    df_result = df_copy.loc[indices_to_keep].drop(columns=[ColumnNames.BASE_NAME]).reset_index(drop=True)
    return df_result

def _download_image(url, save_path):
    """
    Downloads an image from a given URL and saves it to the specified file path.
    Args:
        url (str): The URL of the image to download.
        save_path (str): The file path where the downloaded image will be saved.
    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP request.
        Exception: If there is an error while saving the image.
    Notes:
        - The function sends a GET request with custom headers to mimic a browser request.
        - The image is saved in the format determined by the file extension in `save_path`.
        - Prints a success message if the image is saved successfully, or an error message otherwise.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1',  # Do Not Track request header
        }

        # Send GET request with headers
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check for successful request (status code 200)

        # Open and save the image
        image = Image.open(BytesIO(response.content))
        image.save(save_path)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {e}")
    except Exception as e:
        print(f"Error saving the image: {e}")

FITZPATRICK_IMAGE_HASHES = [ '00f1b7bb9581e91250a2bb224c3321b8', '0ea04f0f86df742d1ab4850a61c888ba', '0fe16f60d42c04ba809fd70085a11b50',
                             '10cbd07a7f2e658a27f31eac2ae17269', '13aa2de8804ae601cd45c1c4cac9bc6f', '184d4c82d2d715ccc299db11357ea24c', 
                             '1a9e904ebc92a8322d12a24a79b02b61', '1c5adfdbc1c9bda0bc57c328a0ed4b35', '1cf28107bbd82eebc64c7c395d711c2b', 
                             '1d76114ea1cee496eb21836e03bb6aca', '1d78d91cb196cb45b40c615cbf7a94ff', '23b55a045c5f88159ce54f69df645c5f', 
                             '25f0286ffedff3a8a17bcdb8f220cdaf', '263049401830de0dee7571422e407e47', '296ec4776e89a65ce8e262ed545ed4bb', 
                             '29a86d89608e63ccfc57b1e40c827ead', '2bceb0364ef64fe3ef5236db2cfbe02a', '2d8dd8e2b1d3f0af318b320f7d77e595', 
                             '2d924a5a87ebe16ae171aa19ae304f5b', '2f269eeeca0c01edeaae2f7cac656a6e', '32f210990987f39829733c1ab5e7f307', 
                             '3a99a5259471f5fee1d4db0d2067b413', '407d41dc4b0e9e0fddd2fe5a4e155986', '46a52b0203c2e2e10adcfdb6dd3d30a0', 
                             '47cb1bb512da0f22002f75e9f158c6e8', '4981320ef16be15730f7fce6207c9f88', '4abcc083990e6408c7936f204a913a81', 
                             '4caae357b8c7b4bc6085936bbaa10740', '4ef4e764a2ae0de2bd5d15affdf8cc20', '4ff1ea06e98ef27ae93b27fbfa384a75', 
                             '543017d891174b7fe9443a4dfa537410', '593c544bf75684c898826405ab9b3971', '5d07648c809dcaca09feef5c856ed3ed', 
                             '5f789231cfe92cef1737d0da94bfdad2', '62def66062d6648a61d4003c4b8ddfd2', '688ac2ba9aff76f589c14be2813f9d5f', 
                             '68daecb39b330760c4faddc47e842c77', '6cf10bfb15d43652310bd5ce509069a9', '6ed58ed2e4c8b378493cd5766f7bad56', 
                             '70a0b051fb01abcd5cdd01bf8b6aa5da', '70febe1909055262ca282551a222a10f', '76a5da87f72f5712243ff4bcf52337b6', 
                             '794572175ce2487d073bcbd2d62f2743', '7a2e6969eb645f2aced2aaa7b309b573', '7c015dcdbe6e3f3c2911a64342c135c6', 
                             '80602e43414e592443aee9e67fcca5c7', '85eecdee71e8ac56831eb6bf276c28bb', '88585fc7b3ebb360c2ca8bc2cdfaf736', 
                             '8972c76e23d9f6d7c301532573961b99', '8ad1a12672261156e2dd3ac462215d76', '8cb3945cf58f8e5540208408a50b5827', 
                             '961154706f5938c81e64757ae78fa2d5', '98280785c17860f879d21525a92af31f', '98b0322096085a163b8b849ec481e30f', 
                             '98ebd0568dc1804989f83706c229b5b7', '998ab155036c1e9bee491e7f41eb629b', '9d83cf9b9cc4a03580de63b62ea2a8f0', 
                             '9e9056fd822eb40e875da5573b284c7e', 'a13eaf2826329cee2e8ab3b16a9df3f9', 'a18693f7fc1cf91848857da05da1efb6', 
                             'a1e1e7987bed1732f1fa29b1945e6cf0', 'a53466176187c41d414de8c9874bd4ce', 'a72ce770400ce35db667f4cef6295192', 
                             'ac82b001c4a7158056790b014728b5e9', 'acb264b31d70f98301b812a134879bbf', 'b024e423de295d22a5f5504c396b6325', 
                             'b02895e0d729c9bcecaed2ee070cada9', 'b50a802b6d5e458e6d6e446b6cee35d9', 'b71c0b85580e06b347da7e16d1aff3ff', 
                             'babe125be18f7ba4fa8a2f7eafce4693', 'be813befb6f1f53ce07c77838af159ea', 'becab807bbe933d1a5e84f378dfc083f', 
                             'c1e1779aa5f1b9afd1016bb8018b5605', 'd0eea0ab5d1d2a33efea4e0257a03825', 'd70a1677a855b45e2049e3f08f47f153', 
                             'd799c42032899b30e39a5ce95c27f098', 'db0585772571cb1d0bb61d1d13f6a052', 'db871411820979160cb098e6453399ce', 
                             'dec2d1ae044897a3eae71fae8a9bc154', 'e64bc4a3e3dfc2db301f0b8c64253da5', 'e668f37499adaead040d7d893e778cc7', 
                             'ea98d1de050f9dc7f3dec15022fa51e5', 'f1e26485eb0dc75866b8526a00b90edd', 'fb37df7e989131ce7a858ec25320067d', 
                             'fd1fa18a3033459631670d50b9b5bfc9', '0ee7316d0dd4ad139cc108e871b7676d', '1dcc2fb45c366dafbb1f45c0e8afd3bd', 
                             '2427f4ddad999d44139aa8b91e937885', '257b98dac6241fa7fc474d3ccd0cc23e', '5b552fb15839338227880aaac895128a', 
                             '5ff2e2cbcbb9f92610ec54dca324d149', '700be0e05d30d1b16be5bf971bba3411', '7041e0d82e0d554237bd1e6dc886a81f', 
                             '79aaed87b0076030d5e6893bff164476', '7e6838d693e1748adf0ea38d6420f8ee', '849d862189a58f44a063ee6ba1a80b1b', 
                             '88c8c110ade2657fbe1c840dd2494511', '89867feafae04841becf921893861722', '915d7fa381560d5b77f1e37cd1c37206', 
                             '9209a4be6a83500ae0c536cc16386284', '97ae91111b8028ee3ce13d6d2bbe2fc5', '9a7cea383c596b7969f945a20ca2f4d1', 
                             '9b6a29c58a5cc9dd856128b9e28881ae', '9f982f2c55254d90155975744427c786', 'a036718d77f4a4339b2aaf15e2087fe0', 
                             'a4df5fa2bab38ee98922e0fbc586b93f', 'a9f949191444e250146b49ff8c95c6ac', 'b09233673fc585369e723ec841ed0acb', 
                             'b3d94ddf1388ed3894ad18ea3fefc880', 'b7ff367d07ee32101f5188a34cc142ab', 'bba3db5954f7975d59a37347222d954d', 
                             'c1cde759eca19dd8fbe3884a352ba3c1', 'd13e9416a0204d2e004f93eb5f853448', 'd4862d1b866e2c536dde339f60de4f0d', 
                             'e19700bdfef1034b79b79f1146d38bca', 'e315766377e07b3a78c789f1ef6cf529', 'e54daa4b483474920f81739c264c1efc', 
                             'e99ba7397c33ba169192ffdb25b66ccf', '008c8be183aa54af2d29a6f75521ebce', '454e68bf5705eabf0f4d28fc1dbd77be', 
                             '53f2eadf5f293c90bb3cda5fb91ec1c5', '6b174a5db8261270d5620a19f78898c8', '7cedaa3277cb1efd41f5f9f70e6be0c7', 
                             'a246a3150fe89472eb84a7d49841ae27', 'de2a6626fa354f736fae29f7402956f1', 'e815861f140e305baf441814e6dbda48', 
                             'f51aae6b3d3b76b975070baac2145672']
