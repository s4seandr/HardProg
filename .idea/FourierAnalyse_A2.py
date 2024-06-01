import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
import tracemalloc

# tracemalloc starten
tracemalloc.start()

# WAV-Datei einlesen
sample_rate, data = wavfile.read('nicht_zu_laut_abspielen.wav')

# Blockgröße und Verschiebung festlegen
block_size = 256  # Beispielwert, Sie können die Blockgröße anpassen
shift = 1  # Verschiebung um 1 Sample

# Liste für die gespeicherten Spektren und Speichernutzung
spectra = []
speicher_nutzung = []

# Schleife über das Signal in Blöcken
for i in range(0, len(data[:, 0]) - block_size, shift):
    # Aktuellen Speicherverbrauch messen
    current, peak = tracemalloc.get_traced_memory()
    speicher_nutzung.append(current / (1024**2))  # Umrechnung von Bytes in MB

    # Extrahieren Sie das aktuelle Blocksignal
    block = data[i:i+block_size, 0]

    # FFT für den Block berechnen
    yf = fft(block)
    xf = fftfreq(block_size, 1 / sample_rate)

    # Speichern Sie das Spektrum
    spectra.append(np.abs(yf[:block_size//2]))

# tracemalloc stoppen
tracemalloc.stop()

# Umwandeln der Liste in ein Numpy-Array für die Darstellung
spectra = np.array(spectra)

# Zeit in Sekunden für die x-Achse berechnen
zeit_in_sekunden = (np.arange(spectra.shape[0]) * shift) / sample_rate

# Zeitachse für den Spektrogramm-Plot anpassen
zeitachse_spektrogramm = np.linspace(0, len(data[:, 0]) / sample_rate, spectra.shape[0])

# Plot des Spektrogramms
plt.figure(figsize=(10, 5))
plt.imshow(np.log(spectra.T + 1e-100), aspect='auto', cmap='viridis', origin='lower', extent=[zeitachse_spektrogramm.min(), zeitachse_spektrogramm.max(), 0, sample_rate / 2])
plt.title('Spektrogramm')
plt.ylabel('Frequenz [Hz]')
plt.xlabel('Zeit [s]')
plt.colorbar(label='Log-Amplitude')
plt.xticks(np.arange(0, len(data[:, 0]) / sample_rate, 10))
plt.savefig('spektrogramm_win_PC.png')  # Speichern des Spektrogramms als PNG

# Erstellen des Graphen für die Speichernutzung in MB
plt.figure(figsize=(10, 5))
plt.plot(zeit_in_sekunden, speicher_nutzung)
plt.title('Speichernutzung über die Zeit in MB')
plt.xlabel('Zeit [s]')
plt.ylabel('Speicherverbrauch (MB)')
max_speicher = max(speicher_nutzung)
plt.figtext(0.3, 0.01, f'Maximaler Speicher: {max_speicher:.2f} MB', ha='center')
gesamtdauer = zeit_in_sekunden[-1]
plt.figtext(0.7, 0.01, f'Gesamtdauer: {gesamtdauer:.2f} s', ha='center')
plt.xticks(np.arange(0, gesamtdauer, 10))
plt.savefig('speichernutzung_win_PC.png')  # Speichern der Speichernutzung als PNG