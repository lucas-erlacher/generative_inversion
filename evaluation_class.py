# this class performs the final evaluation of both our models and the baseline models 

import torch.nn.functional as F
from frechet_audio_distance import FrechetAudioDistance
from tqdm import tqdm 
import librosa
import soundfile as sf
import os
import global_objects
import torch

class EvaluationClass:
    eval_set_size = 100

    # pretrained indicates whether we are evaluating a pretrained baseline-model or a model that we trained ourselves
    def __init__(self, model, final_eval_dataloader, pretrained=False):
        self.model = model
        self.final_eval_dataloader = final_eval_dataloader
        self.pretrained = pretrained

    def eval(self, train_class):
        # compute final eval metrics
        metrics = dict() 
        frechet = FrechetAudioDistance(model_name="vggish", sample_rate=16000, use_pca=False, use_activation=False, verbose=False)
        sum_mse = 0
        sum_cos_sim_row = 0
        sum_cos_sim_col = 0
        frechet_y = []
        frechet_y_hat = []
        counter = 0
        with torch.no_grad():  # this prevents the memory from overflowing in the loop
            for i, batch in tqdm(enumerate(self.final_eval_dataloader), total=self.eval_set_size):
                if i == self.eval_set_size:
                    break
                counter += 1
                x = None
                y = None
                y_hat = None
                loss = None
                if not self.pretrained:
                    x, y, y_hat, loss = train_class.prepare_call_loss(batch, diffusion_generate=True)  # we are dealine with one of our models
                else:
                    self.model.forward(batch)  # we are dealing with a pretrained baseline model
                ####  MSE  ####
                reduced_loss = torch.mean(loss, dim=0)  # need to reduce over batch dim
                sum_mse += reduced_loss
                ####  COS SIM  ####
                sim = F.cosine_similarity(y, y_hat, dim=1)
                normd_sim = self.normalize_cos_sim(sim)
                reduced_sim = torch.mean(normd_sim, dim=0)  # need to reduce over batch dim
                reduced_sim = torch.mean(reduced_sim, dim=0)  # need to reduce over rows
                sum_cos_sim_row += reduced_sim
                sim = F.cosine_similarity(y, y_hat, dim=2)
                normd_sim = self.normalize_cos_sim(sim)
                reduced_sim = torch.mean(normd_sim, dim=0)  # need to reduce over batch dim
                reduced_sim = torch.mean(reduced_sim, dim=0)  # need to reduce over cols
                sum_cos_sim_col += reduced_sim
                ####  FRECHET  ####
                for i in range(x.shape[0]):
                    frechet_y.append(y[i].detach().cpu().numpy())
                    frechet_y_hat.append(y_hat[i].detach().cpu().numpy())
        ####  FRECHET  ####
        # preprocess data
        curr_path = "/itet-stor/elucas/net_scratch/generative_inversion/train_classes/"
        if os.path.exists(curr_path + "/frechet_y"):
            os.system("rm -rf " + curr_path + "/frechet_y")
            os.system("rm -rf " + curr_path + "/frechet_y_hat")
        os.mkdir(curr_path + "/frechet_y")
        os.mkdir(curr_path + "/frechet_y_hat")
        for i in range(len(frechet_y)):
            self.prepare_for_frechet(frechet_y[i], curr_path + "/frechet_y", i)
            self.prepare_for_frechet(frechet_y_hat[i], curr_path + "/frechet_y_hat", i)
        frechet_score = frechet.score(curr_path + "/frechet_y", curr_path + "/frechet_y_hat", dtype="float32")
        
        metrics["mse"] = (sum_mse / counter).item()
        metrics["row_wise_cos_sim"] = (sum_cos_sim_row / counter).item()
        metrics["rol_wise_cos_sim"] = (sum_cos_sim_col / counter).item()
        metrics["frechet_audio_dist"] = frechet_score

        # write metrics to txt file
        with open(train_class.checkpoints_path + "/metrics.txt", "w") as f:
            for key, value in metrics.items():
                f.write(str(key) + ": " + str(value) + "\n")

    # little hleper function that transforms the ragne of cos sim from [-1, 1] to [0, 1]
    def normalize_cos_sim(self, cos_sim):
        return (cos_sim + 1) / 2

    # the frechet library expects the data in a certain format, this method prepares the data for that
    def prepare_for_frechet(self, data, base_path, file_counter):
        data = global_objects.stft_system.invert_spectrogram(data)  # convert to waveform
        # resample data
        resampled_data = librosa.resample(data, orig_sr=global_objects.config["sampling_rate"], target_sr=16000)
        # save the data
        sf.write(base_path + "/" + str(file_counter) + ".wav", resampled_data, 16000)
        return resampled_data