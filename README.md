# VGGish+TCN

This repository containes the code for reproducing the VGGish+TCN results for the ComParE2022 challenge.

First you need to install the dependencies. To do that, run:

`pip install -r requirements.txt`

The `utils` directory contains scripts for preparing the data.

`extract_vggish.py` extracts features from raw audio files.

`prepare_data.py` formats the data in a JSON format needed for SpeechBrain.

The `pre-trained_model` directory contains checkpoints of the for the pre-trained models. To change the model that you want to use, you need to set the proper path of the `output_folder` in the `hyperparamters.yaml` file.

To run the experiments, use:

`python train.py hyperparameters.yaml`
