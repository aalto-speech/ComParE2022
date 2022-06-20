import torch
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.nnet.losses import nll_loss
from speechbrain.utils.checkpoints import Checkpointer

import sentencepiece as spm
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, confusion_matrix
from hyperpyyaml import load_hyperpyyaml
import os
import sys
import json
import numpy as np
import tqdm
from torchvggish import vggish, vggish_input


global criterion
criterion = torch.nn.NLLLoss(reduction="mean", weight=torch.tensor([0.0054, 0.0012, 0.0048, 0.0032, 0.0014, 0.0059, 0.0188, 0.0192]).cuda())


class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        #"Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        audio_embeddings, lens = batch.sig
        encoder_out = self.modules.encoder(audio_embeddings)
        
        # Output layer for sentiment prediction
        predictions = {"sentiment_logprobs": self.hparams.log_softmax(encoder_out)}
        
        return predictions, lens

    
    def compute_objectives(self, predictions, batch, stage):
        # Compute NLL loss
        predictions, lens = predictions
        sentiments, sentiment_lens = batch.sentiments_encoded
        
       
        loss = criterion(
            predictions["sentiment_logprobs"],
            sentiments.squeeze(-1) 
        )

        if stage != sb.Stage.TRAIN:
            prediction_logprobs = predictions["sentiment_logprobs"]
            topi, topk = prediction_logprobs.topk(1)

            self.accuracy_metric.append(predictions["sentiment_logprobs"].unsqueeze(1), sentiments, sentiment_lens)
        
        return loss

    
    def on_stage_start(self, stage, epoch=None):
       # Set up statistics trackers for this stage
        if stage != sb.Stage.TRAIN:
            self.accuracy_metric = self.hparams.accuracy_computer()


    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["ACC"] = self.accuracy_metric.summarize()


        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            ) 
            
            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"]}, max_keys=["ACC"],
                num_to_keep=1
            )


        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            
            with open(self.hparams.decode_text_file, "w") as fo:
                for utt_details in self.wer_metric.scores:
                    print(utt_details["key"], " ".join(utt_details["hyp_tokens"]), file=fo)
   
            
    
    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        
        ckpts = self.checkpointer.find_checkpoints(
                max_key=max_key,
                min_key=min_key,
        )
        model_state_dict = sb.utils.checkpoints.average_checkpoints(
                ckpts, "model" 
        )
        self.hparams.model.load_state_dict(model_state_dict)


    def is_ctc_active(self, stage):
        """Check if CTC is currently active.
        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        """
        if stage != sb.Stage.TRAIN:
            return False
        current_epoch = self.hparams.epoch_counter.current


        return current_epoch <= self.hparams.number_of_ctc_epochs
    

    def evaluate_data(
            self,
            dataset, # Must be obtained from the dataio_function
            max_key, # We load the model with the highest ACC
            loader_kwargs, # opts for the dataloading
        ):

        # If dataset isn't a Dataloader, we create it. 
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )

        self.checkpointer.recover_if_possible(max_key=max_key)
        self.modules.eval() # We set the model to eval mode (remove dropout etc)

        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():
            true_sentiments = []
            pred_sentiments = []
            filenames_array = []
            pred_prob = []
            #for batch in tqdm(dataset, dynamic_ncols=True):
            for batch in dataset:
                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied 
                # in compute_forward(). 

                out = self.compute_forward(batch, stage=sb.Stage.TEST) 
                predictions, wav_lens = out

                # extract features
                #prediction_embedding = predictions["sentiment_logits"].cpu().detach().numpy()
                filenames = batch.id
                
                # Sentiment prediction
                prediction_logprobs = predictions["sentiment_logprobs"]
                topi, topk = prediction_logprobs.topk(1)
                topi = predictions["sentiment_logprobs"].cpu().detach().numpy()

                topk = topk.cpu().detach().numpy().flatten()

                sentiments, sentiment_lens = batch.sentiments_encoded
                sentiments = sentiments.squeeze()

                sentiments = sentiments.cpu().detach().numpy()
               
                true_sentiments.append(sentiments)
                pred_sentiments.append(topk)
                filenames_array.append(filenames)
        
                
                for i in range(topi.shape[0]):
                    pred_prob.append(topi[i])
        
        true_sentiments = np.concatenate(true_sentiments)
        pred_sentiments = np.concatenate(pred_sentiments)
        filenames_array = np.concatenate(filenames_array)
        pred_prob = np.array(pred_prob)
        pred_prob = pred_prob[:, np.newaxis, :]
        
        
        #with open("output/true_devel.txt", "w") as f:
        #    for i in range(len(true_sentiments)):
        #        f.write(filenames_array[i] + "," + str(true_sentiments[i]))
        #        f.write("\n")    

        #with open("output/predictions_devel.csv", "w") as f:
        #    for i in range(len(pred_sentiments)):
        #        f.write(filenames_array[i] + "," + str(pred_sentiments[i]))
        #        f.write("\n")

        #with open("predictions.txt", "w") as f:
        #    for i in pred_sentiments:
        #        f.write(str(i))
        #        f.write("\n")    
       
        #np.save("output/predicted_tcn.npy", pred_sentiments)
        #np.save("output/true_tcn.npy", true_sentiments)
        #np.save("output/probs_test_tcn.npy", np.exp(pred_prob))
        
        cm = confusion_matrix(true_sentiments, pred_sentiments)
        print(cm)

        print("UAR: ", balanced_accuracy_score(true_sentiments, pred_sentiments))
        print("ACC: ", accuracy_score(true_sentiments, pred_sentiments))


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "train_vggish.json"), replacements={"data_root": data_folder})
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "devel_vggish.json"), replacements={"data_root": data_folder})
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "devel_vggish.json"), replacements={"data_root": data_folder})

    datasets = [train_data, valid_data, test_data]
    

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(file_path):
        sig = np.load(file_path, allow_pickle=True)
        #sig = sb.dataio.dataio.read_audio(file_path)
        
        if len(sig.shape) == 1:
            sig = sig[np.newaxis, :]
        
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("sentiment")
    @sb.utils.data_pipeline.provides("sentiments_encoded")
    def text_pipeline(sentiment):
        sentiments_encoded = hparams["sentiment_encoder"].encode_sequence_torch([sentiment])
        yield sentiments_encoded

    
    
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    
    hparams["sentiment_encoder"].update_from_didataset(train_data, output_key="sentiment")

    # save the encoder
    #hparams["sentiment_encoder"].save(hparams["sentiment_encoder_file"])
    
    # load the encoder
    hparams["sentiment_encoder"].load_if_possible(hparams["sentiment_encoder_file"])

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "sentiments_encoded"])
    
    #train_data = train_data.filtered_sorted(sort_key="length", reverse=False)
    
    return train_data, valid_data, test_data




def main(device="cuda"):
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    sb.utils.distributed.ddp_init_group(run_opts) 
    
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
     

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        )


    # Dataset creation
    train_data, valid_data, test_data = data_prep("../../../../Stuttering/data", hparams)
    
    # Training/validation loop
    if hparams["skip_training"] == False:
        print("Training...")
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    
    else:
        # evaluate
        print("Evaluating")
        asr_brain.evaluate_data(test_data, "ACC", hparams["test_dataloader_options"])


if __name__ == "__main__":
    main()
