# this script runs the evaluation of the pretrained baseline models

from evaluation_class import EvaluationClass
from loader import Loader
import sys
sys.path.append("..")
from BigVGAN.inference import load_model as load_bigvgan
from hifiGAN.inference import load_model as load_hifigan
from models.baseline import SimpleModel
from diffusers import AudioLDM2Pipeline

if __name__ == "__main__":
    models = []

    '''

    # set sr to 16000
    # ...

    # load audioLDm
    pipeline = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music")
    vocoder = pipeline.vocoder
    models.append((vocoder, "audioldm2"))

    pipe = pipeline.to("cuda")
    prompt = "Techno music with a strong, upbeat tempo and high melodic riffs."
    audio = pipe(prompt, num_inference_steps=200, audio_length_in_s=1.0).audios[0]

    # set sr to 22050
    # ...
    '''

    # load simple_baseline
    model = SimpleModel()
    models.append((model, "simple_baseline"))

    # load HifiGAN
    checkpoint_path = "/itet-stor/elucas/net_scratch/VCTK_V1"
    model = load_hifigan(checkpoint_path)
    models.append((model, "hifigan"))

    # load bigVGAN
    checkpoint_path = "/itet-stor/elucas/net_scratch/bigvgan_22khz_80band"
    model = load_bigvgan(checkpoint_path)
    models.append((model, "bigvgan"))

    print("loaded models")

    for model in models:
        print()
        print("#" * 20)
        print("evaluating model", model[1])
        print("#" * 20)
        print()
        loader = Loader()
        final_eval_dataloader = loader.get_test_loader(batch_size=1) # batch size can be anything here since it does not change the result of the evaluation
        eval_class = EvaluationClass(model=model[0], final_eval_dataloader=final_eval_dataloader, pretrained=True, name=model[1])
        eval_class.eval(None)