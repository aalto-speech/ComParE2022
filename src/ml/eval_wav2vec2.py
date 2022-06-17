from datasets import load_dataset, Dataset, load_metric, Audio
import torch
from transformers import pipeline, AutoFeatureExtractor, AutoTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, AutoModelForAudioClassification, TrainingArguments, Trainer
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
from ci import CI
import os
import sys

categories = {'Modified': 0, 'no_disfluencies': 1, 'Prolongation': 2, 'SoundRepetition': 3, 'Fillers': 4, 'Garbage': 5, 'Block': 6, 'WordRepetition': 7, '?': -1}
id2cat = {}
for k in categories:
    id2cat[categories[k]] = k
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    #f1_score = f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")
    accuracy = accuracy_score(eval_pred.label_ids, predictions)
    uar = recall_score(eval_pred.label_ids, predictions, average='macro')
    #spearmanr = spearmanr_metric.compute(predictions=predictions, references=eval_pred.label_ids)
    #return {"f1": f1_score, "spearmanr": spearmanr}
    return {"uar": uar, "accuracy": accuracy}
    #return accuracy_score

def prepare_example(example): 
    if '.FI0' in example["file"]:
        example["speech"], example["sampling_rate"] = sf.read(example["file"], channels=1, samplerate=16000, format='RAW', subtype='PCM_16')
    else:
        example["audio"], example["sampling_rate"] = librosa.load(example["file"], sr=16000)
    example["duration_in_seconds"] = len(example["audio"]) / 16000
    return example

def preprocess_function(examples):
    audio_arrays = examples["audio"]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate
    )
    return inputs

def map_to_array(example):
    speech, _ = librosa.load(example["file"], sr=16000, mono=True)
    example["speech"] = speech
    return example

def map_to_pred(batch):
    input_values = processor(batch["audio"], sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values.to(device)).logits
    batch["probs"] = torch.softmax(logits, dim=-1)
    predicted_ids = torch.argmax(logits, dim=-1)
    batch["predictions"] = predicted_ids
    return batch


path_to_recs = "dist/wav/"
freeze_feature_extractor=True
batch_size = 10
model_link = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_link, cache_dir="wav2vec2_cache/" )
processor =  Wav2Vec2Processor.from_pretrained(model_link, cache_dir="wav2vec2_cache/" )
model_link = sys.argv[1]
model = AutoModelForAudioClassification.from_pretrained(
    model_link,
    cache_dir = "wav2vec2_cache",
    num_labels=8
)
if freeze_feature_extractor:
    model.freeze_feature_extractor()
model.to(device)
label_base = "dist/lab"
labels = pd.concat([pd.read_csv(f"{label_base}/{partition}.csv") for partition in ["train", "devel", "test"]])

train_ids = pd.read_csv(f"{label_base}/train.csv").filename
dev_ids = pd.read_csv(f"{label_base}/devel.csv").filename
test_ids = pd.read_csv(f"{label_base}/test.csv").filename
my_dict_train = {'file': [path_to_recs+item for item in train_ids],
        'label': [categories[labels[labels.filename==item].label.item()] for item in train_ids]}
my_dict_dev = {'file': [path_to_recs+item for item in dev_ids],
        'label': [categories[labels[labels.filename==item].label.item()] for item in dev_ids]}
my_dict_test = {'file': [path_to_recs+item for item in test_ids],
        'label': [categories[labels[labels.filename==item].label.item()] for item in test_ids]}
#train_dataset = Dataset.from_dict(my_dict_train)
dev_dataset = Dataset.from_dict(my_dict_dev)
test_dataset = Dataset.from_dict(my_dict_test)

#train_dataset = train_dataset.map(prepare_example, remove_columns=['file'])
dev_dataset = dev_dataset.map(prepare_example, remove_columns=['file'])
test_dataset = test_dataset.map(prepare_example, remove_columns=['file'])

#train_dataset = train_dataset.map(preprocess_function, batched=True, batch_size=1)
#test_dataset = test_dataset.map(preprocess_function, batched=True, batch_size=1)
#dev_dataset = dev_dataset.map(preprocess_function, batched=True, batch_size=1)

prior = np.histogram(my_dict_train['label'], 8)[0]
print(prior)
prior = prior/np.sum(prior)
print(prior)
#Eval dev and test:
print("Devel:\n")
result_folder = sys.argv[2]
result_dev = dev_dataset.map(map_to_pred)
prob_dev = result_dev["probs"]
np.save(os.path.join(result_folder, "probs_devel.npy"), prob_dev)
result_dev = result_dev["predictions"]
result_dev = [i[0] for i in result_dev]
ref = dev_dataset['label'] 
cm = confusion_matrix(ref, result_dev)
acc = accuracy_score(ref, result_dev)
uar = recall_score(ref, result_dev, average='macro')
ci_low, ci_high = CI(np.array(result_dev), np.array(ref))

print("Confusion matrix: ")
print(cm)
print(f"UAR: {uar:.2%} ({ci_low:.2%} - {ci_high:.2%}), ACC: {acc:.2%}") # 
prob_dev = np.array(prob_dev)
result_dev = prob_dev.reshape(prob_dev.shape[0], prob_dev.shape[2])/prior
print(result_dev.shape)
result_dev = np.argmax(result_dev, axis = 1)
print(result_dev.shape)
cm = confusion_matrix(ref, result_dev)
acc = accuracy_score(ref, result_dev)
uar = recall_score(ref, result_dev, average='macro')
ci_low, ci_high = CI(np.array(result_dev), np.array(ref))

print("Alternative Confusion matrix: ")
print(cm)
print(f"Alternative UAR: {uar:.2%} ({ci_low:.2%} - {ci_high:.2%}), ACC: {acc:.2%}")

#cr_devel = classification_report(devel_y, devel_preds)
#mega_report_devel = pycm.ConfusionMatrix(devel_y, devel_preds)
#print(cr_devel)
#write csv file:
pred_dev = [id2cat[val] for val in result_dev]
ref_lab = [id2cat[val] for val in ref]

df_predictions_devel = pd.DataFrame({"filename": dev_ids.to_list(), "prediction": pred_dev, "true": ref_lab})
df_predictions_devel.to_csv(os.path.join(result_folder, "predictions_devel.csv"), index=False)
print("dev predictions saved to "+result_folder +"predictions_devel.csv")
result_test = test_dataset.map(map_to_pred)
prob_test = result_test["probs"]
np.save(os.path.join(result_folder, "probs_test.npy"), prob_test)
result_test = result_test["predictions"]
result_test = [i[0] for i in result_test]
pred_test = [id2cat[val] for val in result_test]
df_predictions_devel = pd.DataFrame({"filename": test_ids.to_list(), "prediction": pred_test})
df_predictions_devel.to_csv(os.path.join(result_folder, "predictions_test.csv"), index=False)
print("test predictions saved to "+result_folder +"predictions_test.csv")

