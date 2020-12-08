Transfer learning
=================

The present decade has made leaps in the area of Deep Learning research, making automating many arduous tasks. However, these highly accurate models require weeks of training on large cloud clusters/GPU machines that consume a lot of compute resources. But, there is a silver lining; the AI community has been open-source from the start with most of the model details and trained parameters available for use.

In the area of NLP, one can train and fine-tune their language models using the [Hugging Face](https://huggingface.co/) library. In this article, we apply transfer learning to an Image Classification task using pre-trained model weights and fine-tuning to our case from start to end. The steps are:

## STEP 1: Analyze the approach

1. Trained model selection

This is the most important part of one's research: to determine the best-fit scenario for our application. For example, a CNN model trained for **Adversarial networks** might not be able to solve an **Image Classification** task with the same high accuracy. Thus, a thorough research on type of dataset the models were trained on and similarity with our problem is necessary to select the best candidate.

2. Training approach

The following guide helps in finalizing our approach

![](assets/transfer-learning-guide.png)

In the case of Computer Vision tasks, if the data is:
- SMALL and SIMILAR: Fine-tune last layers
- SMALL and DIFFERENT: Fine-tune initial layers that learns low-level features like curves
- LARGE and SIMILAR: Fine-tune last layers or the entire network
- LARGE and DIFFERENT: Retrain from the trained model's checpoint

Similarly, in the case of NLP tasks, we can utilize the pretrained word embeddings and train the rest based on our model architecture. 
In addition, the approach can also be determined based on the compute resources and time available.

## STEP 2: Preparing the dataset

Alongwith the model research, understanding of the input is very essential to help us pre-process the data for the pretrained model. Lets try an Image Classification task using PyTorch.

```python
import os, os.path as osp
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2 
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Tensor
from torch.nn import MSELoss, Linear, ReLU, Sigmoid, Module, BCELoss
from torch.optim import SGD, Adam
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

We get a dataframe with info of our data called `train_df` with 2 columns:
- **filename**: of all images
- **target**: respective target labels (ex: cat, dog, etc.) from our dataset

Let's create an encoder for the target variable

```python
from sklearn import preprocessing

categories = train_df.targets.unique()

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(categories)
```

Next is the part of creating a basic Dataset for training the model. The main requirements are the `__len__` and `__getitem__` functions and we can add other functions for convenience.

```python
class ImageDataset(torch.utils.data.Dataset):
    'Generates data for PyTorch'
    def __init__(self, folder):
        self.folder = folder
        self.files = os.listdir(self.folder)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
       
        file = self.files[index]

        # X
        image_loc = os.path.join(self.folder, file)
        image = cv2.imread(image_loc)
        image_tensor = torchvision.transforms.functional.to_tensor(image)

        ''' Additional transforms can be made if data input doesn't match model input layer like:

        transform=transforms.Compose([
            SquarePad(),
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image_tensor)
        '''

        # Y
        file_ID = int(file[:-4]) # Filename w/o extension
        y = label_encoder.transform(train_df.loc[file_ID].values)
        y = torch.tensor(y[0])  

        return image_tensor, y
```

## STEP 3: 
Notable Research

- In 2017, a highly accurate model for [predicting breast cancer](https://www.nature.com/articles/nature21056.epdf) was fine-tuned on Google’s Inception v3 CNN architecture. 

