import torch
from torch import nn
from torchvision import models
import argparse

import config

device = torch.device('cpu')

def convert_model(model, batch_size, hparams):



    # model surgery (model dependent)
    #model = models.resnet18()
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, len(class_names))
    model.set_swish(memory_efficient=False)
    model = nn.Sequential(model, nn.Softmax(dim=1))

    # Input to the model
    x = torch.randn(batch_size, 3, hparams.input_size, hparams.input_size, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,                     # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      args["model"].replace(args["model"].split(".")[-1],'onnx'),   # where to save the model (can be a file or file-like object)
                      input_names = ['input'],   # the model's input names
                      output_names = ['classLabelProbs'], # the model's output names
                      verbose=False)



if __name__ == '__main__':
    """
    Converts pytorch classification model to onnx format.

    Command:
        python convert2onnx.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str,
                    default="trained_models/model.pt", help="Path to model.pt or model.ckpt")
    ap.add_argument("-b", "--batch_size", type=int,
                    default=1, help="Batch size")
    args = vars(ap.parse_args())


    from lightning_model import LightningClassifier
    model = LightningClassifier(config.hparams)
    model.load_state_dict(torch.load(args["model"]))
    model.eval()


    convert_model(model.model, args["batch_size"], config.hparams)
