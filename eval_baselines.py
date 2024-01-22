# this script runs the evaluation of the pretrained baseline models

from evaluation_class import EvaluationClass
from loader import Loader

model_names = ["bigVGAN", "HifiGAN"]

if __name__ == "__main__":
    for model in model_names:
        # load model
        
        loader = Loader()
        final_eval_dataloader = loader.get_test_loader(batch_size=1) # batch size can be anything here since it does not change the result of the evaluation
        eval_class = EvaluationClass(model=model, final_eval_dataloader=final_eval_dataloader, pretrained=True)