from Sound_sensor_test import *
import numpy as np
from scipy.io.wavefile import write
import librosa
import matplotlib.pyplot as plt


#Initializing the sound capturing
main()
print("Sound Capturing function started")

#Processing the sound
sound_array = np.array(sound_captured, dtype=np.int16)
write('waves_captured/captured_sound.wav', 16000, sound_array)
print("saved")
y, sr = librosa.load('waves_captured/captured_sound.wav', sr=None)
mel_spec=librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#display this spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='coolwarm')
plt.title('Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
#have not changed n_fft and hop_length, will try it with these first
