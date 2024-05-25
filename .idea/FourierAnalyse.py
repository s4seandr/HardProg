import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import stft

# WAV-Datei einlesen
sample_rate, data = wavfile.read('Geheimnisvolle_Wellenlaengen.wav')

# Blockgröße und Verschiebung festlegen
block_size = 1024  # Beispielwert, Sie können die Blockgröße anpassen
shift = 32  # Verschiebung um 1 Sample

# Berechnung der STFT für die Daten
f, t, Zxx = stft(data[:, 0], fs=sample_rate, nperseg=block_size, noverlap=block_size-shift)

# Plot des Spektrogramms
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis_r')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.show()