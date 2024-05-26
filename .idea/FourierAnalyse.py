import tracemalloc
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

# Starten Sie das Tracing des Speichers
tracemalloc.start()

# WAV-Datei einlesen
sample_rate, data = wavfile.read('nicht_zu_laut_abspielen.wav')

# Blockgröße und Verschiebung festlegen
block_size = 2048  # Beispielwert, Sie können die Blockgröße anpassen
shift = 4  # Verschiebung um 1 Sample

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

    # Stoppen Sie das Tracing und holen Sie sich die aktuelle, spitzen und gesamte Größe
    current, peak = tracemalloc.get_traced_memory()

    # Geben Sie die Speichernutzung aus
    print(f"Aktuelle Speichernutzung: {current / 10**6}MB; Spitzenverbrauch: {peak / 10**6}MB")

# Umwandeln der Liste in ein Numpy-Array für die Darstellung
spectra = np.array(spectra)

# Plot des Spektrogramms
plt.imshow(np.log(spectra.T + 1e-100), aspect='auto', cmap='viridis', origin='lower')
plt.title('Spektrogramm')
plt.ylabel('Frequenz [Hz]')
plt.xlabel('Zeit [Blöcke]')
plt.colorbar(label='Log-Amplitude')
plt.show()

# Speichern Sie den Plot als PNG-Datei
#plt.savefig('spektrogramm.png')

# Stoppen Sie das Tracing
tracemalloc.stop()