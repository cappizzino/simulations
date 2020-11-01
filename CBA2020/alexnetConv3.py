import os, os.path
import numpy as np
import torch
import torchvision
import torch.nn as nn

from torchvision import transforms
from torchvision import models
from PIL import Image

def main():

    numberFiles = 1
    DIR = "/home/cappizzino/Documentos/doutorado/simulation/paper/dataset"

    for i in range(4):
        open(DIR + "/%d.txt" %i,'w').close()
    
    path_im = [os.path.join(DIR,sp) for sp in [
        'fall/',
        'spring/',
        'summer/',
        'winter/']]

    # Load CNN
    original_model = models.alexnet(pretrained=True)
    class AlexNetConv3(nn.Module):
                def __init__(self):
                    super(AlexNetConv3, self).__init__()
                    self.features = nn.Sequential(
                        # stop at conv3
                        *list(original_model.features.children())[:7]
                    )
                def forward(self, x):
                    x = self.features(x)
                    return x

    model = AlexNetConv3()
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for i in range(4):

        for j in range(numberFiles):
            # sample execution (requires torchvision)
            nameFile = path_im[i] + "%d.png" %j
            input_image = Image.open(nameFile)
            # preprocessing
            input_tensor = preprocess(input_image)
            # create a mini-batch as expected by the model
            input_batch = input_tensor.unsqueeze(0) 
            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')

            with torch.no_grad():
                output = model(input_batch)

            outputFlatten = torch.flatten(output[0])
            outputFlatten = (outputFlatten.data).numpy()
            outputFlatten = outputFlatten / np.linalg.norm(outputFlatten)

        f=open(DIR + "/%d.txt" %i,'a')
        np.savetxt(f,[outputFlatten], delimiter=',', fmt='%1.3f')
        f.close()

if __name__ == "__main__":
    main()