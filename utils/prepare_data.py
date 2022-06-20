import os
from shutil import copy
import json


def prepare_data_vggish(data):
    data = data[1:]
    data_obj = {}
    audio_length = 0 
    counter = 0

    for line in data:
        filename = line.split(",")[0].replace(".wav", "")
        #@file_path = "data/audio/" + filename + ".wav"
        
        file_path_1 = os.path.join(os.getcwd(), "data/Voc_formant_modification/wav_m_f_vggish/", filename + ".npy")
        
        sentiment = line.split(",")[1].rstrip()
        
        counter += 1
        
        data_obj[filename] = {"file_path": file_path_1,
                    "sentiment": sentiment}
        
    # save the json
    with open("data/train_vggish_formant_0.1_m_f.json", 'w') as outfile:
        json.dump(data_obj, outfile)


def prepare_data(data):
    data = data[1:]
    data_obj = {}
    audio_length = 0 
    counter = 0

    for line in data:
        filename = line.split(",")[0].replace(".wav", "")
        #@file_path = "data/audio/" + filename + ".wav"
        
        file_path = os.path.join(os.getcwd(), "data/wav_m_f", filename + ".wav")
        #file_path_1 = os.path.join(os.getcwd(), "data/wav_m_f", filename + "_male.wav")

        sentiment = line.split(",")[1].rstrip()
       
        if "test" in filename:
            data_obj[filename] = {"file_path": file_path,
                        "sentiment": sentiment}
            #data_obj[filename+"_male"] = {"file_path": file_path_1,
            #            "sentiment": sentiment}       
            
       # save the json
    with open("data/train_m_f.json", 'w') as outfile:
        json.dump(data_obj, outfile)



with open("data/lab/train.csv", "r") as f:
    train_data = f.readlines()

with open("data/lab/devel.csv", "r") as f:
    dev_data = f.readlines()

with open("data/lab/test.csv", "r") as f:
    test_data = f.readlines()

prepare_data_vggish(train_data)
