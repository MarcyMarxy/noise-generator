from flask import Flask, render_template, request, jsonify
import numpy as np
import wave
import io
import base64
import numpy as np
import essentia.standard as es
import json

# Initialize the Flask application
app = Flask(__name__)


def generate_noise(
    fs=44100,
    noise_type="uniform",
    duration=0.5,
    mean=0,
    std=1,
    lfo_freq=0,
    adsr_params=None
):
    """
    Generate noise signal.

    :param fs: Sampling rate in Hz, defaults to 44100.
    :type fs: int
    :param noise_type: Generating noise with either a uniform or Gaussian distribution. Uniform noise: Values are uniformly distributed between -1 and 1. Gaussian noise: Values follow a normal distribution with specified mean and standard deviation.
    :type noise_type: str
    :param duration: Duration of the noise in seconds.
    :type duration: float
    :param mean: Mean for Gaussian noise, defaults to 0.
    :type mean: float, optional
    :param std: Standard deviation for Gaussian noise, defaults to 1.
    :type std: float, optional
    :param lfo_freq: LFO frequency in Hz, defaults to 0.
    :type lfo_freq: float, optional
    :param adsr_params: ADSR parameters in seconds, defaults to None.
    :type adsr_params: list[float], optional
    :return: WAV file in memory and generated noise signal (float32 values).
    :rtype: tuple(io.BytesIO, numpy.ndarray)
    """

    samples = int(fs * duration)
    if noise_type == "uniform":
        noise = np.random.uniform(-1, 1, samples)
    elif noise_type == "gaussian":
        noise = np.random.normal(mean, std, samples)
    else:
        raise ValueError("Invalid noise type")

    if lfo_freq > 0:
        noise = apply_lfo(noise, fs, lfo_freq)

    if adsr_params:
        noise = apply_adsr(noise, fs, adsr_params)

    # Pad to even size for FFT compatibility (chatgpt helped me to write this part)
    if len(noise) % 2 != 0:
        noise = np.append(noise, 0)

    # Normalize to 16-bit PCM (chatgpt helped me to write this part)
    noise = (noise * 32767).astype(np.int16)

    # Convert back to float32
    # noise = noise.astype(np.float32)

    # Create WAV file in memory (chatgpt helped me to write this part)
    output = io.BytesIO()
    with wave.open(output, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(noise.tobytes())
    output.seek(0)

    noise = noise.astype(np.float32)
    # print(noise.dtype, "333333333")
    return output, noise


def apply_lfo(noise, fs, lfo_freq):
    """
    Apply loudness low frequency oscillation (LFO) to a signal.

    :param noise: Input signal.
    :type noise: numpy.ndarray
    :param fs: Sampling rate in Hz.
    :type fs: int
    :param lfo_freq: Frequency of the LFO in Hz.
    :type lfo_freq: float
    :return: Signal modulated with LFO.
    :rtype: numpy.ndarray
    """
    lfo = np.sin(2 * np.pi * lfo_freq * np.arange(len(noise)) / fs)
    result = noise * lfo
    return result


def apply_adsr(noise, fs, adsr_params):
    """
    Apply loudness envelope (ADSR) to a signal. (chatgpt helped me to write this part)

    :param noise: Input signal.
    :type noise: numpy.ndarray
    :param fs: Sampling rate in Hz.
    :type fs: int
    :param adsr_params: List containing [attack, decay, sustain, release] times in seconds.
    :type adsr_params: list[float]
    :return: Signal shaped with ADSR envelope.
    :rtype: numpy.ndarray
    """
    attack, decay, sustain, release = adsr_params
    total_samples = len(noise)

    tolerance = 1e-6
    if attack + decay + release >= total_samples / fs + tolerance:
        raise ValueError(
            "Error: attack + decay + release time should be shorter than duration."
        )

    # the lengths of noise and envelope are sometimes off by one. This could be due to rounding issues when calculating the lengths of the different segments of the envelope. To ensure they match, you can adjust the length of the last segment of the envelope to make sure it sums up to the total number of samples
    envelope = np.concatenate(
        [
            np.linspace(0, 1, int(attack * fs)),
            np.linspace(1, sustain, int(decay * fs)),
            np.full(int(total_samples - int((attack + decay + release) * fs)), sustain),
            np.linspace(
                sustain,
                0,
                total_samples
                - (
                    int(attack * fs)
                    + int(decay * fs)
                    + int(total_samples - int((attack + decay + release) * fs))
                ),
            ),
        ]
    )
    # print(len(noise), len(envelope))
    result = noise[: len(envelope)] * envelope
    return result


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
    centroid = es.Centroid(range=fs // 2)(np.abs(np.fft.rfft(signal)))
    log_attack_time = es.LogAttackTime(sampleRate=fs)
    log_attack_time_value = log_attack_time(signal)[0]
    return log_attack_time_value, centroid


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
    Plot clusters and compare generated noise using sound features (chatgpt helped me to write this part).
    
    :param noise: The generated noise data.
    :type noise: numpy.ndarray
    :param noise_label: Label for the generated noise.
    :type noise_label: str
    :param features_dict: Dictionary containing pre-analyzed sound features.
    :type features_dict: dict
    :param fs: Sampling rate in Hz, defaults to 44100.
    :type fs: int, optional
    :return: Data dictionary for the scatter plot.
    :rtype: dict
    """
    # Analyze features of the noise
    log_attack_time_value, spectral_centroid = analyze_features(noise, fs)

    # Extract features for existing clusters
    cluster_log_attack_time_value = np.array(features_dict["log_attack_time_value"])
    cluster_spectral_centroid = np.array(features_dict["spectral_centroid"])
    cluster_labels = np.array(features_dict["label"])
    filenames = np.array(features_dict["filename"])

    # Define color mapping
    color_map = {"noise": "#7e54a7", "sfx": "#28a745", "music": "#5b7fdb"}

    # Prepare data for the scatter plot
    scatter_x = []
    scatter_y = []
    colors = []
    labels = []

    # Add points for existing clusters
    for label in ["noise", "sfx", "music"]:
        indices = np.where(cluster_labels == label)[0]
        scatter_x.extend(cluster_log_attack_time_value[indices])
        scatter_y.extend(cluster_spectral_centroid[indices])
        colors.extend([color_map[label]] * len(indices))
        labels.extend([label] * len(indices))

    # Add point for the generated noise
    scatter_x.append(log_attack_time_value)
    scatter_y.append(spectral_centroid)
    colors.append(
        color_map.get(noise_label, "#d74b73")
    )  # Use 'red' for new noise label
    labels.append(noise_label)
    filenames = filenames.tolist()
    filenames.append("generated noise")

    scatter_x = np.array(scatter_x)
    scatter_y = np.array(scatter_y)
    # updated_labels = [f'{label}\n{filename}' for label, filename in zip(labels, filenames)]

    # Prepare data for the frontend
    data_dict = {
        "x": scatter_x.tolist(),
        "y": scatter_y.tolist(),
        "colors": colors,
        "labels": labels,
        "filenames": filenames,
    }

    return data_dict


@app.route("/")
def index():
    """
    Render the main webpage.

    Route: /

    :return: Rendered HTML template for the index page.
    :rtype: str
    """
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """
    Handle POST requests to generate noise and return audio data.

    Route: /generate

    :return: JSON response containing audio data and features.
    :rtype: flask.Response
    """
    params = request.json
    noise_type = params.get("noiseType")
    duration = float(params.get("duration", 0.5))
    mean = float(params.get("mean", 0))
    std = float(params.get("std", 1))
    lfo_freq = float(params.get("lfoFreq", 0))
    adsr_params = params.get("adsrParams", None)

    if adsr_params:
        adsr_params = [float(x) for x in adsr_params]

    fs = 44100
    audio, noise32 = generate_noise(fs, noise_type, duration, mean, std, lfo_freq=lfo_freq, adsr_params=adsr_params)

    # Encode audio to Base64 for web playback
    audio_base64 = base64.b64encode(audio.read()).decode("utf-8")

    # Feature extraction
    spectrum = es.Spectrum()
    freq_spectrum = spectrum(noise32)

    
    time_axis = np.linspace(0, duration, len(noise32))
    freq_spectrum = np.abs(np.fft.rfft(noise32))
    frequencies = np.fft.rfftfreq(len(noise32), 1 / fs)

    # Plot the generated noise compared to clusters using Essentia features
    features_dict = load_features_from_file(file_path="sound_features.json")
    # plot_cluster_comparison(noise, noise_label=f"Custom Noise", features_dict=features_dict)
    scatterdata = plot_cluster_comparison(
        noise32, noise_label="Custom Noise", features_dict=features_dict
    )

    return jsonify(
        {
            "audioBase64": audio_base64,
            "waveform": {"x": time_axis.tolist(), "y": noise32.tolist()},
            "spectrum": {"x": frequencies.tolist(), "y": freq_spectrum.tolist()},
            "scatter": scatterdata,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

# export FLASK_ENV=development
# export FLASK_DEBUG=1
