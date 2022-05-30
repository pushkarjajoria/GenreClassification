"""
@Time    : 29.11.21 15:57
@Author  : Pushkar Jajoria
@File    : preprocessor.py
@Package : MLwithAudio
"""
import json
import math
import os

import audioread
import librosa
import torch
import tqdm
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

DATASET_PATH = "../dataset/genres_dataset"
JSON_PATH = "../dataset/genre_classificatin_mfcc_labels.json"
SAMPLE_RATE = 22050
DURATION = 30   # Sec
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_len=512, num_segments=5):
    """
    Saves the mfcc coefficients of a dataset into json files to be used later while training the model.
    :param dataset_path: Path to the dataset
    :param json_path: Path to the json file to store it in
    :param n_mfcc: number of MFCC coefficients
    :param n_fft: Window size for each fft
    :param hop_len: Window shift size
    :param num_segments: Since we only have a 100 datasamples for each genre. We use this parameter for
    data augmentation by chopping each track into different segments.
    :return: None
    """
    # dictionary to store data.
    data = {
            "mapping": [],
            "mfcc": [],
            "labels": []
    }

    number_of_samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)
    num_mfcc_per_seg = math.ceil(number_of_samples_per_segment / hop_len)

    # Create a dict from genre -> label_number(int) to use in the MFCC dataset.
    genres_label_dict = {k: v for v, k in enumerate(filter(lambda x: x != ".DS_Store", os.listdir(dataset_path)))}
    data["mapping"] = genres_label_dict

    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):
        # Ensure we are not at the root level
        if dirpath == dataset_path:
            continue
        genre_label = dirpath.split("/")[-1]
        print("")
        print(f"Processing {genre_label}...")

        for f in tqdm.tqdm(filenames):
            file_path = os.path.join(dirpath, f)
            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            except audioread.exceptions.NoBackendError:
                print(f"Unable to process {file_path}. Moving to the next file.")
                continue
            # Process segments extracting MFCC and data
            for s in range(num_segments):
                start_sample = number_of_samples_per_segment * s
                finish_sample = start_sample + number_of_samples_per_segment

                # Store mfcc if it has the expected length to make sure the training data is of the same shape.
                mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                            sr=SAMPLE_RATE, n_mfcc=n_mfcc, n_fft=n_fft,
                                            hop_length=hop_len)
                mfcc = mfcc.T
                if len(mfcc) == num_mfcc_per_seg:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(genres_label_dict[genre_label])

    with open(json_path, "w") as handle:
        json.dump(data, handle, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
