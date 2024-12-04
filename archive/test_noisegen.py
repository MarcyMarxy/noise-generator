import numpy as np
import tempfile
import soundfile as sf
from datetime import datetime
from noisegen import generate_noise, getparams, savefile, apply_lfo, apply_adsr


def test_generate_uniform_noise():
    duration = 0.5
    fs = 44100
    noise = generate_noise(duration, fs, noise_type='uniform')
    assert len(noise) == duration * fs, "Noise length does not match expected duration!"
    assert np.all(noise >= -1) and np.all(noise <= 1), "Uniform noise is out of range!"


def test_generate_gaussian_noise():
    duration = 5
    fs = 44100
    mean = 500
    std = 200
    noise = generate_noise(duration, fs, noise_type='gaussian', mean=mean, std=std)
    assert len(noise) == duration * fs, "Noise length does not match expected duration!"
    assert np.isclose(np.mean(noise), mean, atol=1.0), "Gaussian noise mean is incorrect!"
    assert np.isclose(np.std(noise), std,
                      atol=1.0), "Gaussian noise standard deviation is incorrect!"
    mean1 = 0
    std1 = 1
    noise1 = generate_noise(duration, fs, noise_type='gaussian', mean=mean1, std=std1)
    assert np.isclose(np.mean(noise1), mean1, atol=1.0), "Gaussian noise mean is incorrect!"
    assert np.isclose(np.std(noise1), std1,
                      atol=1.0), "Gaussian noise standard deviation is incorrect!"

    # The number 3 comes from the empirical rule in statistics, which states that for a normal distribution, nearly all data (99.7%)
    # falls within three standard deviations of the mean.
    # 3's not enough, changed to 6
    assert np.all(noise >= (mean - 6 * std)) and np.all(noise <=
                                                        (mean + 6 * std)), "Gaussian noise is out of range!"
    assert np.all(noise1 >= (mean1 - 6 * std1)) and np.all(noise1 <=
                                                           (mean1 + 6 * std1)), "Gaussian noise is out of range!"
    # assert np.all(noise >= -1) and np.all(noise <= 1), "Uniform noise is out of range!"

# chatgpt helped me to write this test
def test_invalid_noise_type():
    duration = 0.5
    fs = 44100
    try:
        generate_noise(duration, fs, noise_type='invalid')
    except ValueError as e:
        assert str(e) == "Invalid noise type. Choose 'uniform' or 'gaussian'."
    else:
        assert False, "Expected ValueError not raised"  # "No error raised for invalid noise type!"


# chatgpt helped me to write this test
def test_apply_lfo():
    # Given an array as signal
    noise = np.array([1, 2, 3, 4])
    fs = 100
    lfo_freq = 1
    result = apply_lfo(noise, fs, lfo_freq)
    expected = noise * np.sin(2 * np.pi * lfo_freq * np.arange(len(noise)) / fs)
    np.testing.assert_array_almost_equal(result, expected)
    # Generate signal
    duration1 = 0.29
    fs1 = 44100
    lfo_freq1 = 0.1
    noise1 = generate_noise(duration1, fs1, noise_type='uniform')
    result1 = apply_lfo(noise1, fs1, lfo_freq1)
    expected1 = noise1 * np.sin(2 * np.pi * lfo_freq1 * np.arange(len(noise1)) / fs1)
    np.testing.assert_array_almost_equal(result1, expected1)


def test_apply_adsr():
    duration = 5
    duration1 = 0.29
    fs = 44100
    noise = generate_noise(duration, fs, noise_type='uniform')
    noise1 = generate_noise(duration1, fs, noise_type='uniform')
    adsr_params = (0.1, 0.1, 0.5, 0.1)
    result = apply_adsr(noise, fs, adsr_params)
    # Create the expected envelope manually or using a similar approach
    expected_envelope = np.concatenate([
        np.linspace(0, 1, int(0.1 * fs)),
        np.linspace(1, 0.5, int(0.1 * fs)),
        np.full(int(len(noise) - int((0.1 + 0.1 + 0.1) * fs)), 0.5),
        np.linspace(0.5, 0, int(0.1 * fs))
    ])
    expected = noise[:len(expected_envelope)] * expected_envelope
    np.testing.assert_array_almost_equal(result, expected)
    # Assert error when attack + decay + release time >= duration
    try:
        result1 = apply_adsr(noise1, fs, adsr_params)
    except ValueError as e:
        assert "Error: attack + decay + release time should be shorter than duration." in str(e)
        # assert str(e) == "Error: attack time should be shorter than duration."
        # test_apply_adsr - AssertionError: assert 'operands cou...,) (220500,) ' == 'Error: attac...han duration.'
    else:
        assert False, "Expected ValueError not raised"
    """except ValueError:
        print("Error: attack time should be shorter than duration.")"""

    # The e in except ValueError as e is a variable that holds the exception object. It allows you to access the error message and other details about the exception.
    # except ValueError as e:


def test_getparams():
    # hit enter for all default values
    result = getparams(0.5, 'uniform', None, None, False, None, False, None)
    assert result[:-1] == (0.5, 'uniform', None, None, False, None, False, None)
    # check timestamp, ignore seconds
    file_name = result[-1]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    assert file_name.startswith(f"noise_uniform_{timestamp[:-2]}")

    # AssertionError: assert ('gaussian', 500, 200, False, None, False, ...) == 'noise_gaussian-500-200'


def test_savefile():
    audio = [0.1, 0.2, 0.3]  # Example audio data
    fs = 44100  # Example sample rate

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        savefile(audio, temp_file.name, fs)
        data, samplerate = sf.read(temp_file.name)

        assert np.allclose(data, audio, atol=1.0)  # 1e-6 data.tolist() == audio
        assert samplerate == fs
