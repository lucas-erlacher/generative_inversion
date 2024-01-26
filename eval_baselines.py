# this script runs the evaluation of the pretrained baseline models

from evaluation_class import EvaluationClass
from loader import Loader
import sys
sys.path.append("..")
from BigVGAN.inference import load_model as load_bigvgan
from hifiGAN.inference import load_model as load_hifigan

if __name__ == "__main__":
    models = []

    # load HifiGAN
    checkpoint_path = "/itet-stor/elucas/net_scratch/VCTK_V1"
    model = load_hifigan(checkpoint_path)
    models.append((model, "hifigan"))

    # load bigVGAN
    checkpoint_path = "/itet-stor/elucas/net_scratch/bigvgan_22khz_80band"
    model = load_bigvgan(checkpoint_path)
    models.append((model, "bigvgan"))

    for model in models:
        loader = Loader()
        final_eval_dataloader = loader.get_test_loader(batch_size=1) # batch size can be anything here since it does not change the result of the evaluation
        eval_class = EvaluationClass(model=model[0], final_eval_dataloader=final_eval_dataloader, pretrained=True, name=model[1])
        eval_class.eval(None)