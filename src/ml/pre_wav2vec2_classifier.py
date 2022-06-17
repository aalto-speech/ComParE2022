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

TRAIN_FINAL = False
categories = {'surprise': 0, 'fear': 1, 'anger': 2, 'pleasure': 3, 'pain': 4, 'achievement': 5, '?': -1}
id2cat = {}
for k in categories:
    id2cat[categories[k]] = k

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_values = feature_extractor(batch["audio"], sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    batch["predictions"] = predicted_ids
    return batch


path_to_recs = "dist/wav/"

model_link = "facebook/wav2vec2-large-xlsr-53"#"voidful/wav2vec2-xlsr-multilingual-56" #"jonatasgrosman/wav2vec2-large-xlsr-53-german"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_link, cache_dir="wav2vec2_cache/" )
#processor =  Wav2Vec2Processor.from_pretrained(model_link, cache_dir="wav2vec2_cache/" )

label_base = "dist/lab"
labels = pd.concat([pd.read_csv(f"{label_base}/{partition}.csv") for partition in ["train", "devel", "test"]])

train_ids = pd.read_csv(f"{label_base}/train.csv").filename
dev_ids = pd.read_csv(f"{label_base}/devel.csv").filename
test_ids = pd.read_csv(f"{label_base}/test.csv").filename
my_dict_train = {'file': [path_to_recs+item for item in train_ids],
        'label': [categories[labels[labels.filename==item].label.item()] for item in train_ids]}
my_dict_dev = {'file': [path_to_recs+item for item in dev_ids],
        'label': [categories[labels[labels.filename==item].label.item()] for item in dev_ids]}

if TRAIN_FINAL:
    my_dict_train['file'] += my_dict_dev['file']
    my_dict_train['label'] += my_dict_dev['label']

my_dict_test = {'file': [path_to_recs+item for item in test_ids],
        'label': [categories[labels[labels.filename==item].label.item()] for item in test_ids]}
train_dataset = Dataset.from_dict(my_dict_train)
dev_dataset = Dataset.from_dict(my_dict_dev)
test_dataset = Dataset.from_dict(my_dict_test)

train_dataset = train_dataset.map(prepare_example, remove_columns=['file'])
dev_dataset = dev_dataset.map(prepare_example, remove_columns=['file'])
test_dataset = test_dataset.map(prepare_example, remove_columns=['file'])

train_dataset = train_dataset.map(preprocess_function, batched=True, batch_size=1)
test_dataset = test_dataset.map(preprocess_function, batched=True, batch_size=1)
dev_dataset = dev_dataset.map(preprocess_function, batched=True, batch_size=1)

model = AutoModelForAudioClassification.from_pretrained(
    model_link,
    cache_dir = "wav2vec2_cache",
    num_labels=6
)

model_name = model_link.split("/")[-1]
freeze_feature_extractor=True
batch_size = 10

args = TrainingArguments(
    "wav2vec2/multiling_pre",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="uar",
    push_to_hub=False,
    gradient_checkpointing=True,
    save_total_limit=2
)

if freeze_feature_extractor:
    model.freeze_feature_extractor()

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)
trainer.train()

#Eval dev and test:
print("Devel:\n")
result_folder = f"./results/wav2vec2_pre/"
os.makedirs(result_folder, exist_ok=True)
result_dev = dev_dataset.map(map_to_pred)["predictions"]
result_dev = [i[0] for i in result_dev]
ref = dev_dataset['label'] 
cm = confusion_matrix(ref, result_dev)
acc = accuracy_score(ref, result_dev)
uar = recall_score(ref, result_dev, average='macro')
ci_low, ci_high = CI(np.array(result_dev), np.array(ref))

print("Confusion matrix: ")
print(cm)
print(f"UAR: {uar:.2%} ({ci_low:.2%} - {ci_high:.2%}), ACC: {acc:.2%}") # 
#cr_devel = classification_report(devel_y, devel_preds)
#mega_report_devel = pycm.ConfusionMatrix(devel_y, devel_preds)
#print(cr_devel)
#write csv file:
pred_dev = [id2cat[val] for val in result_dev]
ref_lab = [id2cat[val] for val in ref]
df_predictions_devel = pd.DataFrame({"filename": dev_ids.to_list(), "prediction": pred_dev, "true": ref_lab})
df_predictions_devel.to_csv(os.path.join(result_folder, "predictions_devel.csv"), index=False)
print("dev predictions saved to "+result_folder +"predictions_devel.csv")
result_test = test_dataset.map(map_to_pred)["predictions"]
result_test = [i[0] for i in result_test]
pred_test = [id2cat[val] for val in result_test]
df_predictions_devel = pd.DataFrame({"filename": test_ids.to_list(), "prediction": pred_test})
df_predictions_devel.to_csv(os.path.join(result_folder, "predictions_test.csv"), index=False)
print("test predictions saved to "+result_folder +"predictions_test.csv")




