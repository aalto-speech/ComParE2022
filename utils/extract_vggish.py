from torchvggish import vggish, vggish_input
import json
import os
import numpy as np


# initialize the VGGish model
embedding_model = vggish()
embedding_model.eval()


data_dir = "data/Voc_formant_modification/wav_m_f"
for filename in os.listdir(data_dir):
    if "train" in filename:
        sig = vggish_input.wavfile_to_examples(os.path.join(data_dir, filename))
        sig = embedding_model.forward(sig)
        sig = sig.detach().numpy()
        
        np.save(os.path.join("data/Voc_formant_modification/wav_m_f_vggish", filename.replace(".wav", "") + ".npy"), sig)
