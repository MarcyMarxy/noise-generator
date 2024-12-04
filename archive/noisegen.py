import numpy as np
import soundfile as sf
from datetime import datetime

def generate_noise(duration, fs, noise_type='uniform', mean=0, std=1):
    """
    Generate basic noise signal.

    :param duration: Duration of the noise in seconds.
    :type duration: float
    :param fs: Sampling rate in Hz.
    :type fs: int
    :param noise_type: Generating noise with either a uniform or Gaussian distribution. Uniform noise: Values are uniformly distributed between -1 and 1. Gaussian noise: Values follow a normal distribution with specified mean and standard deviation.
    :type noise_type: str
    :param mean: Mean for Gaussian noise, defaults to 0.
    :type mean: float, optional
    :param std: Standard deviation for Gaussian noise, defaults to 1.
    :type std: float, optional
    :return: Generated noise signal.
    :rtype: numpy.ndarray
    """
    total_samples = int(duration * fs)
    if noise_type == 'uniform':
        return np.random.uniform(-1, 1, total_samples)
    elif noise_type == 'gaussian':
        return np.random.normal(mean, std, total_samples)
    else:
        raise ValueError("Invalid noise type. Choose 'uniform' or 'gaussian'.")


def apply_lfo(noise, fs, lfo_freq):
    """
    Apply low frequency oscillation (LFO) to a signal.

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
    Apply ADSR envelope to a signal.

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
    try:
        envelope = np.concatenate([
            np.linspace(0, 1, int(attack * fs)),
            np.linspace(1, sustain, int(decay * fs)),
            np.full(int(len(noise) - int((attack + decay + release) * fs)), sustain),
            np.linspace(sustain, 0, int(release * fs))
        ])
        result = noise[:len(envelope)] * envelope
    except ValueError:
        raise ValueError("Error: attack + decay + release time should be shorter than duration.")
        # print("Error: attack + decay + release time should be shorter than duration.")
        # sys.exit()
    return result


def savefile(audio, file_name, fs):
    """
    Save the audio signal to a file.

    :param audio: Input audio signal.
    :type audio: numpy.ndarray
    :param fs: Sampling rate in Hz.
    :type fs: int
    """
    sf.write(file_name, audio, fs)


def getparams(duration=None, noise_type=None, mean=None, std=None, add_lfo=None, lfo_freq=None, add_adsr=None, adsr_params=None): # , add_reverb=None, decay_factor=None, add_reverb=None, decay_factor=None
    """
    Prompt user for noise generation parameters and record the output file name dynamically.

    :return: User-selected parameters.
    :rtype: tuple
    """
    # Prompt user for basic noise generation parameters, initialize a file name.
    duration = float(input("Duration (second, default=0.5): ") or 0.5)
    noise_type = input("Choose noise type (uniform/gaussian)(default=uniform): ").strip().lower() or "uniform"
    file_name = "noise"

    if noise_type == 'gaussian':
        mean = int(input("mean: ").strip())
        std = int(input("std: ").strip())
        file_name += f"_gaussian-{mean}-{std}"
    elif noise_type == 'uniform':
        mean = None
        std = None
        file_name += "_uniform"
    else:
        raise ValueError("Invalid noise type. Choose 'uniform' or 'gaussian'.")

    # Prompt user for lfo parameters. Set default to no.
    add_lfo_input = input("Add LFO modulation (yes/no)? (default=no): ").strip().lower()
    add_lfo = add_lfo_input == "yes" if add_lfo_input else False
    if add_lfo:  # == 'yes'
        lfo_freq = float(input("LFO frequency (Hz, default=1): ") or 1)
        file_name += f"_lfo-{lfo_freq}"
    else:
        lfo_freq = None

    # Prompt user for adsr parameters. Set default to no.
    add_adsr_input = input("Add ADSR envelope (yes/no)? (default=no): ").strip().lower()
    add_adsr = add_adsr_input == "yes" if add_adsr_input else False
    if add_adsr:  # == 'yes'
        adsr_params = input("ADSR parameters (Attack, Decay, Sustain, Release, e.g., 0.01 0.1 0.1 0.3): ").strip() # ,
        adsr_params = list(map(float, adsr_params.split()))
        adsr_params_str = '_'.join(map(str, adsr_params))
        file_name += f"_adsr-{adsr_params_str}"
    else:
        adsr_params = None

    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name += f"_{timestamp}.wav"

    return duration, noise_type, mean, std, add_lfo, lfo_freq, add_adsr, adsr_params, file_name  # , add_reverb, decay_factor, add_delay, delay_time, decay_factor1


def main():
    """
    Main function to generate and process noise.

    Prompts user for parameters, generates noise, applies selected effects,
    and saves the final audio file.
    Sampling rate in Hz, set to fs = 44100.
    """
    fs = 44100
    duration, noise_type, mean, std, add_lfo, lfo_freq, add_adsr, adsr_params, file_name = getparams() # , add_reverb, decay_factor, add_delay, delay_time, decay_factor1

    # Generate noise
    noise = generate_noise(duration, fs, noise_type, mean, std)

    # Apply effects
    if add_lfo:
        noise = apply_lfo(noise, fs, lfo_freq)
    if add_adsr:
        noise = apply_adsr(noise, fs, adsr_params)

    # Save the result
    savefile(noise, file_name, fs)


if __name__ == "__main__":
    main()
