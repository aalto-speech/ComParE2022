from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, RoomSimulator
import numpy as np
import soundfile as sf
import os
import random


augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.6),
    TimeStretch(min_rate=0.7, max_rate=0.99, leave_length_unchanged=False, p=0.6),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.6),
    PitchShift(min_semitones=-4, max_semitones=-1, p=1),
    RoomSimulator(use_ray_tracing=False, p=0.6),
])

for i in range(1, 2):
    for filename in os.listdir("data/Voc_formant_modification/wav_0.4"):
        if "train" in filename:
            audio_data, sr = sf.read(os.path.join("data/Voc_formant_modification/wav_0.4", filename))
            # augment augmentation
            augmented_data = augment(audio_data, sample_rate=sr)
            
            # save the augmented file
            new_filename = filename.replace(".wav", "_" + str(i) + ".wav")
            #new_filename = filename.replace(".wav", "_0.4" + ".wav")

            # save the augmentation
            sf.write(os.path.join("data/Voc_formant_modification/wav_m_f", new_filename), augmented_data, sr)
