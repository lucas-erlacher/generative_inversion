# this script invokes the evaluation of all models (baselines and models trained by me)

from evaluation_class import EvaluationClass
from loader import Loader
import sys
sys.path.append("..")
sys.path.append("../..")
from BigVGAN.inference import load_model as load_bigvgan
from hifiGAN.inference import load_model as load_hifigan
from models.baseline import SimpleModel
import torch
from train_classes import train_class
from models.diffusion_model import DiffusionModel
from models.cnn_1D import Cnn1D
from models.cnn_2D import Cnn2D
from models.unet_1D_wrapper import Unet1DWrapper
from diffusers import DDPMScheduler
import train_invocations.train_diffusion as params
import train_invocations.train_cnn_1D as params_cnn_1D
import train_invocations.train_cnn_2D as params_cnn_2D
import train_invocations.train_unet_1D as params_unet_1D
from train_classes.train_class import TrainClass
import yaml
import time
import global_objects
from pretrained_loader import get_eval_loader

def load():
    models = []

    '''
    # load simple_baseline
    model = SimpleModel()
    models.append((model, "simple_baseline", True))
    
    # load HifiGAN
    checkpoint_path = "/itet-stor/elucas/net_scratch/VCTK_V1"
    model = load_hifigan(checkpoint_path)
    models.append((model, "hifigan", True))


    # load bigVGAN
    checkpoint_path = "/itet-stor/elucas/net_scratch/bigvgan_22khz_80band"
    model = load_bigvgan(checkpoint_path)
    models.append((model, "bigvgan", True))

    # load cnn_1D
    path = "/itet-stor/elucas/net_scratch/generative_inversion/checkpoints/cnn_1D/February08-1922/0_102500.pt"
    model = Cnn1D(params_cnn_1D.kernel_size, params_cnn_1D.pred_diff, params_cnn_1D.in_channels)
    checkpoint = torch.load(path)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to("cuda")
    models.append((model, "cnn_1D", False))

    path = "/itet-stor/elucas/net_scratch/generative_inversion/checkpoints/cnn_2D/February09-1956/0_95500.pt"
    model = Cnn2D(params_cnn_2D.kernel_size, params_cnn_2D.pred_diff)
    checkpoint = torch.load(path)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to("cuda")
    models.append((model, "cnn_2D", False))

    # load unet
    path = "/itet-stor/elucas/net_scratch/generative_inversion/checkpoints/unet_1D/February10-2208/0_106500.pt"
    model = Unet1DWrapper(params_unet_1D.unet_params, params_unet_1D.pred_diff)
    checkpoint = torch.load(path)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to("cuda")
    models.append((model, "unet_1D", False))

    '''

    # diff models
    checkpoints_paths = [
        # "/itet-stor/elucas/net_scratch/generative_inversion/checkpoints/diffusion_model/February11-2219/0_102500.pt",  # diff_run_1
        # "/itet-stor/elucas/net_scratch/generative_inversion/checkpoints/diffusion_model/February12-2217/0_89500.pt",  # diff_run_2
        "/itet-stor/elucas/net_scratch/generative_inversion/checkpoints/diffusion_model/February13-2315/0_147000.pt"  # final diff_run
    ]

    for i, checkpoint_path in enumerate(checkpoints_paths):
        name = checkpoint_path.split("/")[-3]
        name += "_run_" + str(global_objects.config["diff_num_inference_steps"])
        # load model
        scheduler_params = params.scheduler_params
        scheduler = DDPMScheduler(**scheduler_params)  # this is noise scheduler
        model = DiffusionModel(params.unet_params, scheduler)
        # load checkpoint
        checkpoint = torch.load(checkpoint_path)
        # load state dict
        state_dict = checkpoint['model_state_dict']
        # load the state dict into the model
        model.load_state_dict(state_dict)
        model = model.to("cuda")
        models.append((model, name, False))

    print("loaded models")

    return models

if __name__ == "__main__":
    models = load()

    for model in models:
        print()
        print("#" * 20)
        print("evaluating model", model[1])
        print("#" * 20)
        print()
        if model[1] == "hifigan" or model[1] == "bigvgan":
            final_eval_dataloader = get_eval_loader(1)
        else:
            loader = Loader()
            final_eval_dataloader = loader.get_test_loader(batch_size=1) # batch size can be anything here since it does not change the result of the evaluation
        eval_class = EvaluationClass(model=model[0], final_eval_dataloader=final_eval_dataloader, pretrained=model[2], name=model[1])
        if model[2] == False:
            tc = TrainClass(model[0], batch_size=1, eval_freq=1)  # batch_size assumed to be 1 in evaluation_class
            eval_class.eval(tc)
            time.sleep(60) # implementation detail of TrainClass necessitates this
        else:
            eval_class.eval(None)