import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Funktion zur Berechnung der FFT mit variabler Blockgröße für beide Kanäle
def calculate_fft_stereo(data, sample_rate, block_size):
    n = len(data)
    freq = np.fft.rfftfreq(block_size, d=1/sample_rate)
    fft_data_left = np.abs(np.fft.rfft(data[:block_size, 0])) / block_size
    fft_data_right = np.abs(np.fft.rfft(data[:block_size, 1])) / block_size
    return freq, fft_data_left, fft_data_right

# WAV-Datei einlesen
sample_rate, data = wavfile.read('Geheimnisvolle_Wellenlaengen.wav')

# Blockgröße festlegen
block_size = 512  #1024  # Beispielwert, Sie können die Blockgröße anpassen

# FFT für beide Kanäle mit variabler Blockgröße
freq, fft_data_left, fft_data_right = calculate_fft_stereo(data, sample_rate, block_size)

# Plot des Spektrums für beide Kanäle
plt.figure()
plt.plot(freq, fft_data_left, label='Left Channel')
plt.plot(freq, fft_data_right, label='Right Channel')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum with Block Size {} for Stereo Signal'.format(block_size))
plt.legend()
plt.show()