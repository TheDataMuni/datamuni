# Training a T5 Model to Generate Titles for ArXiv Research Papers using Transformers

In this article, you will learn how to train a `T5 model` for text generation - to generate title given a research paper's abstract or summary using **Transformers🤗**

## Introduction
`T5 model` is a Sequence-to-Sequence model. A Sequence-to-Sequence model is fully capable to perform any text to text conversion task. **What does it mean?** - It means that a `T5 model` can take any input text and convert it into any output text. Such text-to-text conversion is useful in NLP tasks like language translation, summarization, text generation etc.

![](https://shivanandroy.com/posts/dl/images/T5.png)

For this tutorial, We will take research paper's abstract or brief summary as our input text and its corrosponding paper's title as output text and feed it to a `T5 model` to train. Once the model is trained, it will be able to generate the paper's title based on the abstract.

So, let's dive in.

## Data
ArXiv has recently open-sourced a monstrous dataset of 1.7M research papers on Kaggle. [Go to Dataset](https://www.kaggle.com/Cornell-University/arxiv). 

We will use its `abstract` and `title` columns to train our model. 
- `title`: This column represents the title of the research paper
- `abstract`: This column represents brief summary of the research paper.

This will be a supervised training where `abstract` is our independent variable `X` while `title` is our dependent variable `y`.

## Let's Code

First, lets install all the dependencies - We will work with latest stable `pytorch 1.6`.

```python
! pip uninstall torch torchvision -y
! pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
! pip install -U transformers
! pip install -U simpletransformers  
```

### Load the Data

The format of the data is a nested `json`
```python
import json

data_file = '../input/arxiv/arxiv-metadata-oai-snapshot.json'

# Helper function to load the dataset
def get_metadata():
    with open(data_file, 'r') as f:
        for line in f:
            yield line
```
```python
# let's see the first row of the data
metadata = get_metadata()
for paper in metadata:
    paper_dict = json.loads(paper)
    print('Title: {}\n\nAbstract: {}\nRef: {}'.format(paper_dict.get('title'), paper_dict.get('abstract'), paper_dict.get('journal-ref')))
#     print(paper)
    break
```

`Title: Calculation of prompt diphoton production cross sections at Tevatron and
  LHC energies`

`Abstract:   A fully differential calculation in perturbative quantum chromodynamics is
presented for the production of massive photon pairs at hadron colliders. All
next-to-leading order perturbative contributions from quark-antiquark,
gluon-(anti)quark, and gluon-gluon subprocesses are included, as well as
all-orders resummation of initial-state gluon radiation valid at
next-to-next-to-leading logarithmic accuracy. The region of phase space is
specified in which the calculation is most reliable. Good agreement is
demonstrated with data from the Fermilab Tevatron, and predictions are made for
more detailed tests with CDF and DO data. Predictions are shown for
distributions of diphoton pairs produced at the energy of the Large Hadron
Collider (LHC). Distributions of the diphoton pairs from the decay of a Higgs
boson are contrasted with those produced from QCD processes at the LHC, showing
that enhanced sensitivity to the signal can be obtained with judicious
selection of events.`

`Ref: Phys.Rev.D76:013009,2007`
***
We have taken 3 attributes of the dataset: `Title`, `Abstract` and `Ref`.
`Ref` is important because the 4 characters of its value give us the year in which the paper was published. 

**We will take last 5 years ArXiv papers (2016-2020)**
```python
titles = []
abstracts = []
years = []
metadata = get_metadata()
for paper in metadata:
    paper_dict = json.loads(paper)
    ref = paper_dict.get('journal-ref')
    try:
        year = int(ref[-4:]) 
        if 2016 < year < 2021:
            years.append(year)
            titles.append(paper_dict.get('title'))
            abstracts.append(paper_dict.get('abstract'))
    except:
        pass 

len(titles), len(abstracts), len(years)
```
`(25625, 25625, 25625)`
***
So, we have around 25K research papers published from 2016 to 2020. Next, we will convert this data into `pandas` dataframe and then we will use this data to train our `T5 model`
```python
papers = pd.DataFrame({
    'title': titles,
    'abstract': abstracts,
    'year': years
})
papers.head()
```
![](https://shivanandroy.com/posts/dl/images/t5-model-dataframe-1.png)

***

### T5 Model Training
We will use `simpletransformers` library to train `T5 model`.

This library is based on the `Transformers` library by HuggingFace. `SimpleTransformers` lets you quickly train and evaluate Transformer models. Only 3 lines of code are needed to initialize a model, train the model, and evaluate a model. You can read more about it here: https://github.com/ThilinaRajapakse/simpletransformers

**Input Data**

`Simpletransformers` implementation of `T5 model` expects a data to be a dataframe with 3 columns: `<prefix>`, `<input_text>`, `<target_text>`

- `<prefix>`: A string indicating the task to perform. (E.g. "question", "stsb", "summarization")
- `<input_text>`: The input text sequence (we will use Paper's abstract as input_text )
- `<target_text>`: The target sequence (we will use Paper's title as output_text )

You can read about the data format: https://github.com/ThilinaRajapakse/simpletransformers#t5-transformer

```python
# Adding <input_text> and <target_text> columns
papers = papers[['title','abstract']]
papers.columns = ['target_text', 'input_text']

# Adding <prefix> columns
papers['prefix'] = "summarize"

# splitting the data into training and test dataset
eval_df = papers.sample(frac=0.2, random_state=101)
train_df = papers.drop(eval_df.index)

train_df.shape, eval_df.shape
```
`((20500, 2), (5125, 2))`
***
We have around 20K research papers for training and 5K papers for evaluation.

**Setting Training Parameters and Start Training**

We will train our `T5 model` with very bare minimum `num_train_epochs=4`, `train_batch_size=16` to fit into Kaggle's compute limits. Feel free to play around these training parameters.
```python
import logging

import pandas as pd
from simpletransformers.t5 import T5Model

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# T5 Training parameters
model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 512,
    "train_batch_size": 16,
    "num_train_epochs": 4,
}

# Create T5 Model
model = T5Model("t5-small", args=model_args, use_cuda=True)

# Train T5 Model on new task
model.train_model(train_df)

# Evaluate T5 Model on new task
results = model.eval_model(eval_df)

print(results)
```
`{'eval_loss': 2.103029722170599}`
***
It took around 4 hours to train for `4 epochs` and with `batch_size` of `16`. And we get a loss of `2.103` on our test data.
## 🥳 Voila! We're Done
Let's see how our model performs in generating paper's titles

**Example 1**
```python
random_num = 350
actual_title = eval_df.iloc[random_num]['target_text']
actual_abstract = ["summarize: "+eval_df.iloc[random_num]['input_text']]
predicted_title = model.predict(actual_abstract)

print(f'Actual Title: {actual_title}')
print(f'Predicted Title: {predicted_title}')
print(f'Actual Abstract: {actual_abstract}')
```
`Actual Title: Cooperative Passive Coherent Location: A Promising 5G Service to Support
  Road Safety`

`Predicted Title: ['CPCL: a distributed MIMO radar service for public users']`

`Actual Abstract: ['summarize:   5G promises many new vertical service areas beyond simple communication and\ndata transfer. We propose CPCL (cooperative passive coherent location), a\ndistributed MIMO radar service, which can be offered by mobile radio network\noperators as a service for public user groups. CPCL comes as an inherent part\nof the radio network and takes advantage of the most important key features\nproposed for 5G. It extends the well-known idea of passive radar (also known as\npassive coherent location, PCL) by introducing cooperative principles. These\nrange from cooperative, synchronous radio signaling, and MAC up to radar data\nfusion on sensor and scenario levels. By using software-defined radio and\nnetwork paradigms, as well as real-time mobile edge computing facilities\nintended for 5G, CPCL promises to become a ubiquitous radar service which may\nbe adaptive, reconfigurable, and perhaps cognitive. As CPCL makes double use of\nradio resources (both in terms of frequency bands and hardware), it can be\nconsidered a green technology. Although we introduce the CPCL idea from the\nviewpoint of vehicle-to-vehicle/infrastructure (V2X) communication, it can\ndefinitely also be applied to many other applications in industry, transport,\nlogistics, and for safety and security applications.\n']`

***

**Example 2**
```python
random_num = 478
actual_title = eval_df.iloc[random_num]['target_text']
actual_abstract = ["summarize: "+eval_df.iloc[random_num]['input_text']]
predicted_title = model.predict(actual_abstract)

print(f'Actual Title: {actual_title}')
print(f'Predicted Title: {predicted_title}')
print(f'Actual Abstract: {actual_abstract}')
```
`Actual Title: Test Model Coverage Analysis under Uncertainty`

`Predicted Title: ['Probabilistic aggregate coverage analysis for model-based testing']`

`Actual Abstract: ['summarize:   In model-based testing (MBT) we may have to deal with a non-deterministic\nmodel, e.g. because abstraction was applied, or because the software under test\nitself is non-deterministic. The same test case may then trigger multiple\npossible execution paths, depending on some internal decisions made by the\nsoftware. Consequently, performing precise test analyses, e.g. to calculate the\ntest coverage, are not possible. This can be mitigated if developers can\nannotate the model with estimated probabilities for taking each transition. A\nprobabilistic model checking algorithm can subsequently be used to do simple\nprobabilistic coverage analysis. However, in practice developers often want to\nknow what the achieved aggregate coverage, which unfortunately cannot be\nre-expressed as a standard model checking problem. This paper presents an\nextension to allow efficient calculation of probabilistic aggregate coverage,\nand moreover also in combination with k-wise coverage.\n']`

***

**Example 3**
```python
random_num = 999
actual_title = eval_df.iloc[random_num]['target_text']
actual_abstract = ["summarize: "+eval_df.iloc[random_num]['input_text']]
predicted_title = model.predict(actual_abstract)

print(f'Actual Title: {actual_title}')
print(f'Predicted Title: {predicted_title}')
print(f'Actual Abstract: {actual_abstract}')
```
`Actual Title: Computational intelligence for qualitative coaching diagnostics:
  Automated assessment of tennis swings to improve performance and safety`

`Predicted Title: ['Personalized qualitative feedback for tennis swing technique using 3D video']`

`Actual Abstract: ['summarize:   Coaching technology, wearables and exergames can provide quantitative\nfeedback based on measured activity, but there is little evidence of\nqualitative feedback to aid technique improvement. To achieve personalised\nqualitative feedback, we demonstrated a proof-of-concept prototype combining\nkinesiology and computational intelligence that could help improving tennis\nswing technique utilising three-dimensional tennis motion data acquired from\nmulti-camera video. Expert data labelling relied on virtual 3D stick figure\nreplay. Diverse assessment criteria for novice to intermediate skill levels and\nconfigurable coaching scenarios matched with a variety of tennis swings (22\nbackhands and 21 forehands), included good technique and common errors. A set\nof selected coaching rules was transferred to adaptive assessment modules able\nto learn from data, evolve their internal structures and produce autonomous\npersonalised feedback including verbal cues over virtual camera 3D replay and\nan end-of-session progress report. The prototype demonstrated autonomous\nassessment on future data based on learning from prior examples, aligned with\nskill level, flexible coaching scenarios and coaching rules. The generated\nintuitive diagnostic feedback consisted of elements of safety and performance\nfor tennis swing technique, where each swing sample was compared with the\nexpert. For safety aspects of the relative swing width, the prototype showed\nimproved assessment ...\n']`

***

The results are absolutely stunning. The generated reserach papers title are exactly human like. That's the power of `T5 Model`!

## Notebooks
- [Go to Dataset](https://www.kaggle.com/Cornell-University/arxiv)
- [Go to Published Kaggle Kernel](https://www.kaggle.com/officialshivanandroy/transformers-generating-titles-from-abstracts)

***