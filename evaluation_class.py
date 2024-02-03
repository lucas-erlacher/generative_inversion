# this class performs the final evaluation of both our models and the pretrained baseline models 

import torch.nn.functional as F
from frechet_audio_distance import FrechetAudioDistance
from tqdm import tqdm 
import librosa
import soundfile as sf
import os
import global_objects
import torch

class EvaluationClass:
    eval_set_size = 2

    # pretrained indicates whether we are evaluating a pretrained baseline-model or a model that we trained ourselves
    def __init__(self, model, final_eval_dataloader, pretrained=False, name=None):
        self.model = model
        self.final_eval_dataloader = final_eval_dataloader
        self.pretrained = pretrained
        self.name = name

    def eval(self, train_class):
        # compute final eval metrics
        metrics = dict() 
        frechet = FrechetAudioDistance(model_name="vggish", sample_rate=16000, use_pca=False, use_activation=False, verbose=False)
        sum_mse = 0
        sum_l1 = 0
        sum_cos_sim_row = 0
        sum_cos_sim_col = 0
        sum_cos_sim_flattened = 0
        sum_waveform_dist_baseline = 0
        sum_waveform_dist = 0
        frechet_y = []
        frechet_y_hat = []
        counter = 0
        with torch.no_grad():  # this prevents the memory from overflowing in the loop
            for i, batch in tqdm(enumerate(self.final_eval_dataloader), total=self.eval_set_size):
                if i == self.eval_set_size:
                    break
                counter += 1
                x, y = batch

                ########  FORWARD PASS  ########
                y_hat = None
                loss = None
                if not self.pretrained:  # the non pretrained models have a method called prepare_call_loss
                    x, y, y_hat, loss = train_class.prepare_call_loss(batch, diffusion_generate=True)  # we are dealine with one of our models
                else:  # the pretrained models only have the forward method
                    if not self.name == "simple_baseline":  # all the baselines that are not simple_baseline return a waveform
                        # audioldm2 needs special treatment
                        if self.name == "audioldm2":
                            x = x[:, :, ::4]  # audioldm2 internally upsamples the time dimension of the spec by a factor of 4
                            x.permute(0, 2, 1)  # the vocoder has time dim first and then mel dim
                        y_hat = self.model.forward(x)  # we are dealing with a pretrained baseline model
                        y_hat_spec = torch.zeros(y.shape)
                        for i in range(y_hat.shape[0]):
                            y_hat_spec[i] = torch.tensor(global_objects.stft_system.spectrogram(y_hat.squeeze(1)[i].detach().cpu().numpy()))
                        y_hat = y_hat_spec   
                    else:  # simple_baseline returns a spec
                        y_hat = self.model.forward(x)
                    loss = F.mse_loss(y_hat, y)

                ########  COMPUTE METRICS  ########
                ####  MSE  ####
                reduced_loss = torch.mean(loss, dim=0)  # need to reduce over batch dim
                sum_mse += reduced_loss
                ####  L1  ####
                l1 = F.l1_loss(y, y_hat)
                sum_l1 = torch.mean(l1, dim=0)  # need to reduce over batch dim
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
                # flattened
                y_flat = y.reshape(y.shape[0], -1) 
                y_hat_flat = y_hat.reshape(y_hat.shape[0], -1)
                sim = F.cosine_similarity(y_flat, y_hat_flat, dim=1)
                normd_sim = self.normalize_cos_sim(sim)
                reduced_sim = torch.mean(normd_sim, dim=0)  # need to reduce over batch dim
                sum_cos_sim_flattened += reduced_sim
                ####  FRECHET  ####
                for i in range(x.shape[0]):
                    frechet_y.append(y[i].detach().cpu().numpy())
                    frechet_y_hat.append(y_hat[i].detach().cpu().numpy())
                ####  WAVEFORM DIST  ####
                # baseline
                for i in range(x.shape[0]):
                    spec = torch.tensor(global_objects.stft_system.spectrogram(global_objects.stft_system.invert_spectrogram(y[i].detach().cpu().numpy())))
                    if not self.pretrained:
                        spec = spec.to("cuda")
                    sum_waveform_dist_baseline += F.mse_loss(spec, y[i])
                # normal waveform dist
                for i in range(x.shape[0]):
                    spec = torch.tensor(global_objects.stft_system.spectrogram(global_objects.stft_system.invert_spectrogram(y_hat[i].detach().cpu().numpy())))
                    if not self.pretrained:
                        spec = spec.to("cuda")
                    sum_waveform_dist += F.mse_loss(spec, y[i])
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
        metrics["l1"] = (sum_l1 / counter).item()
        metrics["row_wise_cos_sim"] = (sum_cos_sim_row / counter).item()
        metrics["rol_wise_cos_sim"] = (sum_cos_sim_col / counter).item()
        metrics["flattened_cos_sim"] = (sum_cos_sim_flattened / counter).item()
        metrics["frechet_audio_dist"] = frechet_score
        metrics["waveform_dist_baseline"] = (sum_waveform_dist_baseline / counter).item()
        metrics["waveform_dist"] = (sum_waveform_dist / counter).item()

        # write metrics to txt file
        if self.pretrained == False:
            path = train_class.checkpoints_path + "/metrics.txt"
        else:
            path = "/itet-stor/elucas/net_scratch/generative_inversion/" + "metrics_" + self.name + ".txt"
        with open(path, "w") as f:
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