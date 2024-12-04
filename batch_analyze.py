import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import essentia.standard as es
import json
from datetime import datetime


def analyze_features(signal, fs):
    """
    Analyze audio features using Essentia.

    :param signal: Input signal.
    :type signal: numpy.ndarray
    :param fs: Sampling rate in Hz.
    :type fs: int
    :return: log_attack_time_value and centroid of the signal.
    :rtype: tuple(float, float)
    """
    signal = signal.astype(np.float32)
    # print(signal.dtype)  # Check the dtype
    # print("Before Centroid:", signal.dtype)

    centroid = es.Centroid(range=fs // 2)(np.abs(np.fft.rfft(signal)))

    # print("Before LogAttackTime:", signal.dtype)
    log_attack_time = es.LogAttackTime(sampleRate=fs)
    log_attack_time_value = log_attack_time(signal)[0]

    # TODO: Other descriptors
    # RMS rms = es.RMS()(noise_signal)
    # spectral contrast, dissonance mean, mfcc mean

    return log_attack_time_value, centroid


def analyze_and_save_folder_features(folder_path, label, features_dict, fs=44100):
    """
    Analyze all sounds in a folder and save their features to a dictionary.
    
    :param folder_path: Path to the folder (noise, sfx, music) containing sounds.
    :type folder_path: str
    :param label: Label (noise, sfx, music) of the audio to add to the features dictionary.
    :type label: str
    :param features_dict: Initialized features dictionary containing keys for storing analyzed sound features.
    :type features_dict: dict
    :param fs: Sampling rate in Hz, defaults to 44100.
    :type fs: int, optional
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            audio, _ = sf.read(file_path)
            # audio = audio[:,0] stereo 2 mono
            audio = audio.astype(np.float32)
            # print(audio.dtype,"1111111111")
            log_attack_time_value, spectral_centroid = analyze_features(audio, fs)
            features_dict["log_attack_time_value"].append(log_attack_time_value)
            features_dict["spectral_centroid"].append(spectral_centroid)
            features_dict["label"].append(label)
            features_dict["filename"].append(filename)


def save_features_to_file(features_dict, file_path="sound_features.json"):
    """
    Save analyzed sound features to a JSON file.
    
    :param features_dict: Loaded dictionary containing pre-analyzed sound features.
    :type features_dict: dict
    :param file_path: Output file path.
    :type file_path: str
    """
    with open(file_path, "w") as f:
        json.dump(features_dict, f)


def load_features_from_file(file_path="sound_features.json"):
    """
    Load sound features from a JSON file.
    
    :param file_path: File path of the JSON file containing pre-analyzed sound features.
    :type file_path: str
    :return: Loaded dictionary containing pre-analyzed sound features.
    :rtype: dict
    """
    with open(file_path, "r") as f:
        features_dict = json.load(f)
    return features_dict



def plot_cluster_comparison(noise, noise_label, features_dict, fs=44100):
    """
    Save a feature comparison plot between clusters and the generated noise (the sample sound in this case)(chatgpt helped me to write this part).
    
    :param noise: The generated noise data.
    :type noise: numpy.ndarray
    :param noise_label: Label for the generated noise.
    :type noise_label: str
    :param features_dict: Dictionary containing pre-analyzed sound features.
    :type features_dict: dict
    :param fs: Sampling rate in Hz, defaults to 44100.
    :type fs: int, optional
    """
    noise = noise.astype(np.float32)
    # print(noise.dtype,"222222222")
    log_attack_time_value, spectral_centroid = analyze_features(noise, fs)

    plt.figure(figsize=(10, 6))

    # Plot the existing clusters (three categories)
    for label, color in zip(["noise", "sfx", "music"], ["purple", "green", "blue"]):
        indices = [i for i, l in enumerate(features_dict["label"]) if l == label]
        plt.scatter(
            np.array(features_dict["log_attack_time_value"])[indices],
            np.array(features_dict["spectral_centroid"])[indices],
            c=color,
            label=label,
            alpha=0.5,
        )

    # Plot the generated noise
    plt.scatter(
        log_attack_time_value,
        spectral_centroid,
        c="red",
        label=f"{noise_label}",
        marker="x",
        s=100,
    )

    plt.title("Feature Comparison: Generated Noise vs Clusters")
    plt.xlabel("log_attack_time_value")
    plt.ylabel("Spectral Centroid")
    plt.legend()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(timestamp + "_PairScatterPlot.png")
    plt.close()


def main():
    """
    Main function to generate, analyze and compare the noise.
    """
    # Initialize the features dictionary to store the features
    features_dict = {
        "log_attack_time_value": [],
        "spectral_centroid": [],
        "label": [],
        "filename": [],
    }

    # Define paths to the three sound categories
    static_noise_folder = "noise"
    synthesized_sound_folder = "sfx"
    music_folder = "music"

    # Analyze and save features from each folder
    analyze_and_save_folder_features(static_noise_folder, "noise", features_dict)
    analyze_and_save_folder_features(synthesized_sound_folder, "sfx", features_dict)
    analyze_and_save_folder_features(music_folder, "music", features_dict)

    # Save the features to a file
    save_features_to_file(features_dict)

    # Load features from the file (when plotting)
    features_dict = load_features_from_file()

    # Read a sample sound
    noise_path = "droneStab-mono.wav"
    noise, fs = sf.read(noise_path)

    # Plot the generated noise compared to clusters
    plot_cluster_comparison(
        noise, noise_label="Generated Noise", features_dict=features_dict
    )
    # noise_label="Generated Noise" not working


if __name__ == "__main__":
    main()
