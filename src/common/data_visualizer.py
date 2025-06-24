import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common.constants import Constants, ColumnNames

def display_basic_analysis(df: pd.DataFrame) -> None:
    """
    Display basic statistics about the dataset (number of samples, class distribution).
    """
    
    benign_samples = df[df[ColumnNames.TARGET] == 0]
    malignant_samples = df[df[ColumnNames.TARGET] == 1]

    print(f"Total number of samples: {len(df)}")
    print(f"Number of benign samples: {len(benign_samples)}")
    print(f"Number of malignant samples: {len(malignant_samples)}")

    # Calculate and print class percentages
    benign_percentage = 0 if len(df) == 0 else len(benign_samples) / len(df) * 100
    malignant_percentage = 0 if len(df) == 0 else len(malignant_samples) / len(df) * 100

    print(f"Percentage of benign samples: {benign_percentage:.2f}%")
    print(f"Percentage of malignant samples: {malignant_percentage:.2f}%")

def display_skin_tone_analysis(df: pd.DataFrame) -> None:
    """
    Display the number of samples for each skin tone class.
    """

    print("\nSkin Tone Analysis:")
    for tone in Constants.ALL_SKIN_TONES:
        count = len(df[df[ColumnNames.SKIN_TONE] == tone])
        print(f"Skin tone {tone}: {count} samples")

def display_category_comparison_chart(df: pd.DataFrame) -> None:
    """
    Display a bar chart comparing benign and malignant sample counts.
    """

    labels = [Constants.BENIGN, Constants.MALIGNANT]
    benign_samples = df[df[ColumnNames.TARGET] == 0]
    malignant_samples = df[df[ColumnNames.TARGET] == 1]
    counts = [len(benign_samples), len(malignant_samples)]

    plt.bar(labels, counts, color=['blue', 'orange'])
    plt.xlabel('Melanoma Type')
    plt.ylabel('Sample Count')
    plt.title('Comparison of Benign and Malignant Samples')
    plt.show()

def display_patient_distribution(df: pd.DataFrame) -> None:
    """
    Display a histogram showing the distribution of samples per patient.
    """

    patient_counts = df['patient_id'].value_counts()

    plt.figure(figsize=(10, 5))
    plt.hist(patient_counts, bins=30, color='purple', edgecolor='black')
    plt.xlabel('Number of Samples per Patient')
    plt.ylabel('Number of Patients')
    plt.title('Patient Distribution by Sample Count')
    plt.show()

def display_top_patients(df: pd.DataFrame) -> None:
    """
    Display a bar chart of the top 30 patients with the most samples.
    """

    top_patients = df[ColumnNames.PATIENT_ID].value_counts().head(30)

    plt.figure(figsize=(12, 6))
    top_patients.plot(kind='bar', color='skyblue')
    plt.xlabel('Patient ID')
    plt.ylabel('Sample Count')
    plt.title('Top 30 Patients with Most Samples')
    plt.xticks(rotation=45)
    plt.show()

def display_least_patients(df: pd.DataFrame) -> None:
    """
    Display a bar chart of the 30 patients with the least samples.
    """

    least_patients = df[ColumnNames.PATIENT_ID].value_counts().tail(30)

    plt.figure(figsize=(12, 6))
    least_patients.plot(kind='bar', color='lightcoral')
    plt.xlabel('Patient ID')
    plt.ylabel('Sample Count')
    plt.title('Bottom 30 Patients with Least Samples')
    plt.xticks(rotation=45)
    plt.show()

def visualize_skin_tone_distribution(df: pd.DataFrame) -> None:
    """
    Visualize the distribution of samples across different skin tone categories.
    """

    tone_counts = df[ColumnNames.SKIN_TONE].value_counts().sort_index()

    # Label mapping
    tone_labels = {
        Constants.VERY_LIGHT_SKIN_TONE: "Very Light",
        Constants.LIGHT_SKIN_TONE: "Light",
        Constants.MEDIUM_LIGHT_SKIN_TONE: "Medium Light",
        Constants.MEDIUM_SKIN_TONE: "Medium",
        Constants.DARK_SKIN_TONE: "Dark"
    }
    labels = [tone_labels.get(tone, str(tone)) for tone in tone_counts.index]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, tone_counts.values)
    plt.title('Skin Tone Distribution')
    plt.xlabel('Skin Tone')
    plt.ylabel('Sample Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 5, f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def visualize_age_distribution(df: pd.DataFrame) -> None:
    """
    Visualize the approximate age distribution of the samples.
    """

    age_series = df[ColumnNames.AGE_APPROX].dropna()

    plt.figure(figsize=(10, 6))
    plt.hist(age_series, bins=10, edgecolor='black', alpha=0.7)
    plt.title('Age Distribution of Patients')
    plt.xlabel('Approximate Age')
    plt.ylabel('Number of Samples')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate counts above bars
    bin_counts, bin_edges = np.histogram(age_series, bins=10)
    for count, edge in zip(bin_counts, bin_edges):
        plt.text(edge + (bin_edges[1] - bin_edges[0]) / 2, count + 2, str(count), ha='center')

    plt.tight_layout()
    plt.show()

def visualize_sex_distribution(df: pd.DataFrame) -> None:
    """
    Visualize the distribution of samples by patient sex.
    """

    sex_counts = df[ColumnNames.SEX].value_counts(dropna=False)
    labels = sex_counts.index.fillna('Unknown').tolist()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, sex_counts.values, color='skyblue')
    plt.title('Sex Distribution')
    plt.xlabel('Sex')
    plt.ylabel('Sample Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 5, f'{int(height)}', ha='center')

    plt.tight_layout()
    plt.show()

def visualize_anatomical_site_distribution(df: pd.DataFrame) -> None:
    """
    Visualize the distribution of samples by anatomical site.
    """

    site_counts = df[ColumnNames.ANATOM_SITE_GENERAL_CHALLENGE].value_counts(dropna=False)
    labels = site_counts.index.fillna('Unknown').tolist()

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, site_counts.values, color='lightcoral')
    plt.title('Anatomical Site Distribution')
    plt.xlabel('Anatomical Site')
    plt.ylabel('Sample Count')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 5, f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
