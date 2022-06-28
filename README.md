# ComParE22
This repository provides the code for running the official baselines and Aalto's systems for ComParE2022.

> Tamás Grósz, Dejan Porjazovski, Yaroslav Getman, Sudarsana Reddy Kadiri, and Mikko Kurimo: "Wav2vec2-based Paralinguistic
Systems to Recognise Vocalised Emotions and Stuttering" in *Proceedings of the 30th International Conference on Multimedia*, (Lisbon, Portugal), ACM, 2022.

```bibtex
@inproceedings{Grosz22ComParE,
author = {Tam\'as Gr\'osz and Dejan Porjazovski and Yaroslav Getman and Sudarsana Reddy Kadiri and Mikko Kurimo},
title = {Wav2vec2-based Paralinguistic Systems to Recognise Vocalised Emotions and Stuttering},
booktitle = {{Proceedings of the 30th International Conference on Multimedia}},
year = {2022},
address = {Lisbon, Portugal},
publisher = {ACM},
month = {October},
note = {to appear},
}
```

## Getting the code
You can find the baseline code and instructions for each sub-challenge on a corresponding branch of this repository. 

To run our **wav2vec2**-based systems, use the `ft_w2v2_classification_compare2022.py` script. 

The VGGish + TCN system can be reproduced using the codes in the **VGGish+TCN branch**. The augmentation script (`utils/data_augmentation.py`) we used in our experiments can also be found in that branch.

## Links to the wav2vec2 models mentioned in the paper

* $wav2vec2_{M}$: https://huggingface.co/facebook/wav2vec2-large-xlsr-53
* $wav2vec2_{M}^{*}$: https://huggingface.co/voidful/wav2vec2-xlsr-multilingual-56
* $wav2vec2_{M}^{de}$: https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-german
* $wav2vec2_{L}^{de}$: https://huggingface.co/aware-ai/wav2vec2-xls-r-1b-5gram-german
* $wav2vec2_{M}^{er}$: https://huggingface.co/superb/wav2vec2-large-superb-er
* $wav2vec2_{S}^{de}$: https://huggingface.co/aware-ai/wav2vec2-base-german
