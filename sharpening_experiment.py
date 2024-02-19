# point of this experiment is to see if applying vertical gradient sharpening can improve the attack of the spectrogram returned by the diff model
# result: does not work, spectrogram sounds unrecognizable

import torch
from loader import Loader
import global_objects
from transforms import spec_to_preprocessed_spec_inverse
from utils_lucas import spec_to_wav
from models.diffusion_model import DiffusionModel
import math
from diffusers import DDPMScheduler
from train_classes.train_class import TrainClass
import train_invocations.train_diffusion as params

model_path = "/itet-stor/elucas/net_scratch/generative_inversion/checkpoints/diffusion_model/February13-2315/0_147000.pt"

if __name__ == "__main__":
    # load data
    loader = Loader()
    eval_loader = loader.get_quick_eval_loader(batch_size=1)

    # load model
    scheduler_params = params.scheduler_params
    scheduler = DDPMScheduler(**scheduler_params)  # this is noise scheduler
    model = DiffusionModel(params.unet_params, scheduler, 1000)
    # load checkpoint
    checkpoint = torch.load(model_path)
    # load state dict
    state_dict = checkpoint['model_state_dict']
    # load the state dict into the model
    model.load_state_dict(state_dict)

    train_class = TrainClass(model, params.batch_size, params.eval_freq)
    # get a sample
    batch = next(iter(eval_loader))
    x, y = batch

    x = x.to('cpu')
    y = y.to('cpu')
    x = train_class.to_starting_point(x)
    x, y = train_class.to_learning_space(x, y)

    x = x.to('cuda')
    y = y.to('cuda')

    x = x.float()
    y = y.float()
    y_hat = model.generate(x)
    x = spec_to_preprocessed_spec_inverse(x[0].detach().cpu().numpy(), numpy=True)

    # apply sharpening
    kernel = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32)
    x = torch.tensor(x, dtype=torch.float32)
    res = torch.nn.functional.conv2d(x.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1).numpy()  # for some reason conv2d convolves
                                                                                                                        # 4 dimensional tensors only ...

    # clip res into 0-1 range
    res = res.clip(0, 1)

    # turn spec into waveform
    spec_to_wav(res[0, 0], "sharpening_experiment_sharpened.wav")