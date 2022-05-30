"""
@Time    : 29.11.21 15:02
@Author  : Pushkar Jajoria
@File    : preprocessing.py
@Package : MLwithAudio
"""
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
file = "./blues.wav"

signal, sr = librosa.load(file)

# Display the waveform
# librosa.display.waveplot(signal, sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()

# FFT
fft = np.fft.fft(signal)
# Move from complex values to get their magnitude
magnitude = np.abs(fft)
freq_bins = np.linspace(0, sr, len(magnitude))

# Why is the plot symmetrical?
# Nyquist-Shannon sampling theorem but as far as we are concerned, we don't need the whole plot.
# The left first half gives us all the information that we need!

left_freq_bins = freq_bins[:int(len(freq_bins)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

plt.plot(left_freq_bins, left_magnitude)
plt.title("FFT")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()


# Short time fourier transform
n_fft = 2048        # Window which are considering when we are performing a single fft transform
hop_length = 512    # Amount are shifting each fft to the right
stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
spectogram = np.abs(stft)
# Humans perceive loudness in a logarithmic scale and let's create a log spectogram
log_spectogram = librosa.amplitude_to_db(spectogram)

librosa.display.specshow(log_spectogram)
plt.title("Short Time Fourier Transform")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()

# MFCC
mfcc = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(mfcc)
plt.title("Mel Frequency Cepstral Coefficients")
plt.xlabel("Time")
plt.ylabel("mfcc coefficients")
plt.colorbar()
plt.show()
