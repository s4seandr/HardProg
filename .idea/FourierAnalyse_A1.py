import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

# WAV-Datei einlesen
sample_rate, data = wavfile.read('nicht_zu_laut_abspielen.wav')

# Blockgröße und Verschiebung festlegen
block_size = 256  # Beispielwert, Sie können die Blockgröße anpassen
shift = 1  # Verschiebung um 1 Sample

# Liste für die gespeicherten Spektren
spectra = []

# Schleife über das Signal in Blöcken
for i in range(0, len(data[:, 0]) - block_size, shift):
    # Extrahieren Sie das aktuelle Blocksignal
    block = data[i:i+block_size, 0]

    # FFT für den Block berechnen
    yf = fft(block)
    xf = fftfreq(block_size, 1 / sample_rate)

    # Speichern Sie das Spektrum
    spectra.append(np.abs(yf[:block_size//2]))

# Umwandeln der Liste in ein Numpy-Array für die Darstellung
spectra = np.array(spectra)

# Zeitachse für den Spektrogramm-Plot anpassen
zeitachse_spektrogramm = np.linspace(0, len(data[:, 0]) / sample_rate, spectra.shape[0])

# Plot des Spektrogramms
plt.figure(figsize=(10, 5))
plt.imshow(np.log(spectra.T + 1e-100), aspect='auto', cmap='viridis', origin='lower', extent=[zeitachse_spektrogramm.min(), zeitachse_spektrogramm.max(), 0, sample_rate / 2])
plt.title('Spektrogramm')
plt.ylabel('Frequenz [Hz]')
plt.xlabel('Zeit [s]')
plt.colorbar(label='Log-Amplitude')

# X-Achse in 10-Sekunden-Intervallen skalieren
plt.xticks(np.arange(0, len(data[:, 0]) / sample_rate, 10))

plt.show()