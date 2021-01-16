# Building A Faster & Accurate Search Engine with Transformers & Haystack


This article will let you build a faster and accurate Search Engine using `Transformers🤗`


![Search Engine](https://shivanandroy.com/posts/dl/images/covid19.jpg)




## Introduction

In this article, we will build a search engine, which will not only **retrieve** and **rank** the articles based on the query but also give us the **response**, along with a 1000 words **context** around the response.

To achieve this, we will need:
 - a structured dataset with reserach papers and its full text.
 - `Transformers` library to build QA model
 - and Finally, `Haystack` library to scale QA model to thousands of documents and build a search engine.

## Data
For this tutorial, we will use [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge). 

*CORD-19 is a resource of over 200,000 scholarly articles, including over 100,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses. This freely available dataset is provided to the global research community to apply recent advances in natural language processing and other AI techniques to generate new insights in support of the ongoing fight against this infectious disease.*

This dataset is ideal for building document retrieval system as it has full research paper content in text format. Columns like 
 - `paper_id`: Unique identifier of research paper
 - `title`: title of research paper
 -  `abstract`: Bried summary of the research paper
 - `full_text`: Full text/content of the research paper

 are of our interest.

In **Kaggle Folder Structure** - There are 2 directories - `pmc_json` and `pdf_json` - which contains the data in the `json` format.
We will take 25,000 articles from `pmc_json` directory and 25000 articles from `pdf_json` - So, a total of 50,000 research articles to build our search engine.

We will extract `paper_id`, `title`, `abstract`, `full_text` and put it in an easy to use `pandas.DataFrame`.


## Let's Code


### Load the data

```python
import numpy as np
import pandas as pd
import os
import json
import re
from tqdm import tqdm


dirs=["pmc_json","pdf_json"]
docs=[]
counts=0
for d in dirs:
    print(d)
    counts = 0
    for file in tqdm(os.listdir(f"../input/CORD-19-research-challenge/document_parses/{d}")):#What is an f string?
        file_path = f"../input/CORD-19-research-challenge/document_parses/{d}/{file}"
        j = json.load(open(file_path,"rb"))
        #Taking last 7 characters. it removes the 'PMC' appended to the beginning
        #also paperid in pdf_json are guids and hard to plot in the graphs hence the substring
        paper_id = j['paper_id']
        paper_id = paper_id[-7:]
        title = j['metadata']['title']

        try:#sometimes there are no abstracts
            abstract = j['abstract'][0]['text']
        except:
            abstract = ""
            
        full_text = ""
        bib_entries = []
        for text in j['body_text']:
            full_text += text['text']
                
        docs.append([paper_id, title, abstract, full_text])
        #comment this below block if you want to consider all files
        #comment block start
        counts = counts + 1
        if(counts >= 25000):
            break
        #comment block end    
df=pd.DataFrame(docs,columns=['paper_id','title','abstract','full_text'])

print(df.shape)
df.head()
```
***
`pmc_json`
 `34%|███▎      | 24999/74137 [01:2902:56, 278.06it/s]`


`pdf_json`
 `25%|██▍       | 24999/100423 [01:24<04:15, 295.09it/s]`

***
`(50000, 4)`

![DataFrame](https://shivanandroy.com/posts/dl/images/covid-search-engine-data.png)

***

We have 50,000 articles and columns like `paper_id`, `title`, `abstract` and `full_text`

We will be interested in `title` and `full_text` columns as these columns will be used to build the engine. Let's setup a Search Engine on top `full_text` - which contains the full content of the research papers.





### Haystack
Now, Welcome `Haystack`! The secret sauce behind setting up a search engine and ability to scale any QA model to thousands of documents.


![HayStack](https://shivanandroy.com/posts/dl/images/haystack1.png)

`Haystack` helps you scale QA models to large collections of documents! You can read more about this amazing library here https://github.com/deepset-ai/haystack

For installation: `! pip install git+https://github.com/deepset-ai/haystack.git`

But just to give a background, there are 3 major components to Haystack.

- **Document Store**: Database storing the documents for our search. We recommend Elasticsearch, but have also more light-weight options for fast prototyping (SQL or In-Memory).
- **Retriever**: Fast, simple algorithm that identifies candidate passages from a large collection of documents. Algorithms include TF-IDF or BM25, custom Elasticsearch queries, and embedding-based approaches. The Retriever helps to narrow down the scope for Reader to smaller units of text where a given question could be answered.
- **Reader**: Powerful neural model that reads through texts in detail to find an answer. Use diverse models like BERT, RoBERTa or XLNet trained via FARM or Transformers on SQuAD like tasks. The Reader takes multiple passages of text as input and returns top-n answers with corresponding confidence scores. You can just load a pretrained model from Hugging Face's model hub or fine-tune it to your own domain data.

And then there is **Finder** which glues together a **Reader** and a **Retriever** as a pipeline to provide an easy-to-use question answering interface.

Now, we can setup `Haystack` in 3 steps:
 1. Install `haystack` and import its required modules
 2. Setup `DocumentStore`
 3. Setup `Retriever`, `Reader` and `Finder` 

### 1. Install `haystack`

Let's install `haystack` and import all the required modules
```python
# installing haystack
! pip install git+https://github.com/deepset-ai/haystack.git

# importing necessary dependencies
from haystack import Finder
from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
```
### 2. Setting up `DocumentStore`

Haystack finds answers to queries within the documents stored in a `DocumentStore`. The current implementations of `DocumentStore` include `ElasticsearchDocumentStore`, `SQLDocumentStore`, and `InMemoryDocumentStore`.

But they recommend `ElasticsearchDocumentStore` because as it comes preloaded with features like full-text queries, BM25 retrieval, and vector storage for text embeddings.

So - Let's set up a `ElasticsearchDocumentStore`.

```python
! wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.2-linux-x86_64.tar.gz -q
! tar -xzf elasticsearch-7.6.2-linux-x86_64.tar.gz
! chown -R daemon:daemon elasticsearch-7.6.2
 
import os
from subprocess import Popen, PIPE, STDOUT
es_server = Popen(['elasticsearch-7.6.2/bin/elasticsearch'],
                   stdout=PIPE, stderr=STDOUT,
                   preexec_fn=lambda: os.setuid(1)  # as daemon
                  )
# wait until ES has started
! sleep 30

# initiating ElasticSearch
from haystack.database.elasticsearch import ElasticsearchDocumentStore
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
```

Once `ElasticsearchDocumentStore` is setup, we will write our documents/texts to the `DocumentStore`.


- Writing documents to `ElasticsearchDocumentStore` requires a format - List of dictionaries as shown below: 
```
[
    {"name": "<some-document-name>, "text": "<the-actual-text>"},
    {"name": "<some-document-name>, "text": "<the-actual-text>"}
    {"name": "<some-document-name>, "text": "<the-actual-text>"}
]
```

(Optionally: you can also add more key-value-pairs here, that will be indexed as fields in Elasticsearch and can be accessed later for filtering or shown in the responses of the Finder)


- We will use `title` column to pass as `name` and `full_text` column to pass as the `text`

```python
# Now, let's write the dicts containing documents to our DB.
document_store.write_documents(data[['title', 'abstract']].rename(columns={'title':'name','full_text':'text'}).to_dict(orient='records'))
```

### 3. Setup `Retriever`, `Reader` and `Finder`

Retrievers help narrowing down the scope for the Reader to smaller units of text where a given question could be answered. They use some simple but fast algorithm.

Here: We use Elasticsearch's default BM25 algorithm
```python
from haystack.retriever.sparse import ElasticsearchRetriever
retriever = ElasticsearchRetriever(document_store=document_store)
```
A Reader scans the texts returned by retrievers in detail and extracts the k best answers. They are based on powerful, but slower deep learning models.

Haystack currently supports Readers based on the frameworks `FARM` and `Transformers`. With both you can either load a local model or one from `Hugging Face's` model hub (https://huggingface.co/models).

Here: a medium sized RoBERTa QA model using a Reader based on FARM (https://huggingface.co/deepset/roberta-base-squad2)

```python
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True, context_window_size=500)
```
`Downloading: 100%`
`1.26k/1.26k [00:33<00:00, 37.9B/s]`

`Downloading: 100%`
`499M/499M [00:22<00:00, 21.8MB/s]`

`Downloading: 100%`
`899k/899k [00:03<00:00, 272kB/s]`

`Downloading: 100%`
`456k/456k [00:01<00:00, 252kB/s]`

`Downloading: 100%`
`150/150 [00:01<00:00, 97.1B/s]`

`Downloading: 100%`
`190/190 [00:00<00:00, 342B/s]`




And finally: The Finder sticks together reader and retriever in a pipeline to fetch answers based on our query.

```python
finder = Finder(reader, retriever)
```
***




### 🥳 Voila! We're Done.
Let's see, how well our search engine works! - For simplicity, we will keep the number of documents to be retrieved to 2 using `top_k_reader` parameter. But we can extend to any number in production.

Now, whenever we search or query our `DocumentStore`, we get 3 responses-
- we get the **answer**
- a 1000 words **context** around the answer
- and the **name/title** of the research paper
***

**Example 1: What is the impact of coronavirus on babies?**

```python
question = "What is the impact of coronavirus on babies?"
number_of_answers_to_fetch = 2

prediction = finder.get_answers(question=question, top_k_retriever=10, top_k_reader=number_of_answers_to_fetch)
print(f"Question: {prediction['question']}")
print("\n")
for i in range(number_of_answers_to_fetch):
    print(f"#{i+1}")
    print(f"Answer: {prediction['answers'][i]['answer']}")
    print(f"Research Paper: {prediction['answers'][i]['meta']['name']}")
    print(f"Context: {prediction['answers'][i]['context']}")
    print('\n\n')
```
`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.17 Batches/s]`

`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.83 Batches/s]`

`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.71 Batches/s]`

`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.75 Batches/s]`

`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.78 Batches/s]`

`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.83 Batches/s]`

`Inferencing Samples: 100%|██████████| 1/1 [00:01<00:00,  1.08s/ Batches]`

`Inferencing Samples: 100%|██████████| 1/1 [00:01<00:00,  1.09s/ Batches]`

`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.83 Batches/s]`

`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.57 Batches/s]`

***
`Question: What is the impact of coronavirus on babies?`

`#1`

`Answer: 
While babies have been infected, the naivete of the neonatal immune system in relation to the inflammatory response would appear to be protective, with further inflammatory responses achieved with the consumption of human milk.`

`Research Paper: COVID 19 in babies: Knowledge for neonatal care`

`Context:
ance to minimize the health systems impact of this pandemic across the lifespan.The Covid-19 pandemic has presented neonatal nurses and midwives with challenges when caring for mother and babies. This review has presented what is currently known aboutCovid-19 and neonatal health, and information and research as they are generated will add to a complete picture of the health outcomes. While babies have been infected, the naivete of the neonatal immune system in relation to the inflammatory response would appear to be protective, with further inflammatory responses achieved with the consumption of human milk. The WHO has made clear recommendations about the benefits of breastfeeding, even if the mother and baby dyad is Covid-19 positive, if they remain well. The mother and baby should not be separated, and the mother needs to be able to participate in her baby's care and develop her mothering role. The complexities of not being able to access her usual support people mean that the mother`

***
***

`#2`
`Answer: 
neonate are mild, with low-grade fever and gastrointestinal signs such as poor feeding and vomiting. The respiratory symptoms are also limited to mild tachypnoea and/or tachycardia.`

`Research Paper: COVID 19 in babies: Knowledge for neonatal care`

`Context: Likewise, if the mother and baby are well, skin-to-skin and breast feeding should be encouraged, as the benefits outweigh any potential harms. If a neonate becomes unwell and requires intensive care, they should be nursed with droplet precautions in a closed incubator in a negative pressure room. The management is dictated by the presenting signs and symptoms. It would appear the presenting symptoms in the neonate are mild, with low-grade fever and gastrointestinal signs such as poor feeding and vomiting. The respiratory symptoms are also limited to mild tachypnoea and/or tachycardia. However, as there has been a presentation of seizure activity with fever a neurological examination should be part of the investigations.As of writing this paper, there has been a preliminary study published in the general media The WHO has also welcomed these preliminary results and state it is "looking forward to a full data analysis" (https://www.who.int/news-room/detail/16-06-2020-who-welcomesprelimin`

***

**Example 2: What is the impact of coronavirus on pregnant women?**

```python
question = "What is the impact of coronavirus on pregnant women?"
number_of_answers_to_fetch = 2

prediction = finder.get_answers(question=question, top_k_retriever=10, top_k_reader=number_of_answers_to_fetch)
print(f"Question: {prediction['question']}")
print("\n")
for i in range(number_of_answers_to_fetch):
    print(f"#{i+1}")
    print(f"Answer: {prediction['answers'][i]['answer']}")
    print(f"Research Paper: {prediction['answers'][i]['meta']['name']}")
    print(f"Context: {prediction['answers'][i]['context']}")
    print('\n\n')
```
`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.17 Batches/s]`

`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.83 Batches/s]`

`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.71 Batches/s]`

`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.75 Batches/s]`

`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.78 Batches/s]`

`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.83 Batches/s]`

`Inferencing Samples: 100%|██████████| 1/1 [00:01<00:00,  1.08s/ Batches]`

`Inferencing Samples: 100%|██████████| 1/1 [00:01<00:00,  1.09s/ Batches]`

`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.83 Batches/s]`

`Inferencing Samples: 100%|██████████| 1/1 [00:00<00:00,  1.57 Batches/s]`

***
`Question: What is the impact of coronavirus on pregnant women?`


`#1`
`Answer: 
pregnant woman may be more vulnerable to severe infection (Favre et al. 2020 ) and evidence from previous viral outbreaks suggests a higher risk of unfavourable maternal and neonatal outcomes in this population`

`Research Paper:
COVID 19 in babies: Knowledge for neonatal care`

`Context: na. The disease manifests with a spectrum of symptoms ranging from mild upper respiratory tract infection to severe pneumonitis, acute respiratory distress syndrome (ARDS) and death.Relatively few cases have occurred in children and neonates who seem to have a more favourable clinical course than other age groups (De Rose et al. 2020) . While not initially identified as a population at risk, pregnant woman may be more vulnerable to severe infection (Favre et al. 2020 ) and evidence from previous viral outbreaks suggests a higher risk of unfavourable maternal and neonatal outcomes in this population (Alfaraj et al. 2019) .Moreover, the associated policies developed as a result of the pandemic relating to social distancing and prevention of cross infection have led to important considerations specific to the field of maternal and neonatal health, and a necessity to consider unintended consequences for both the mother and baby (Buekens et al. 2020) .Countries are faced with a rapidly deve`

***

`#2`
`Answer: 
While not initially identified as a population at risk, pregnant woman may be more vulnerable to severe infection (Favre et al., 2020) and evidence from previous viral outbreaks suggests a higher risk of unfavourable maternal and neonatal outcomes in this population`

`Research Paper: 
COVID 19 in babies: Knowledge for neonatal care`

`Context:
tified in Wuhan, Hubei, China. The disease manifests with a spectrum of symptoms ranging from mild upper respiratory tract infection to severe pneumonitis, acute respiratory distress syndrome (ARDS) and death. Relatively few cases have occurred in children and neonates who seem to have a more favourable clinical course than other age groups (De Rose et al., 2020). While not initially identified as a population at risk, pregnant woman may be more vulnerable to severe infection (Favre et al., 2020) and evidence from previous viral outbreaks suggests a higher risk of unfavourable maternal and neonatal outcomes in this population (Alfaraj et al., 2019). Moreover, the associated policies developed as a result of the pandemic relating to social distancing and prevention of cross infection have led to important considerations specific to the field of maternal and neonatal health, and a necessity to consider unintended consequences for both the mother and baby (Buekens et al., 2020).Countries`

***

The results are meaningful😄. Please note that we have used a pretrained model `deepset/roberta-base-squad2` for this tutorial. We might expect a significant improvement if we use a QA model trained specific to this dataset.

## Notebooks
- [Go to Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
- [Go to Published Kaggle Kernel](https://www.kaggle.com/officialshivanandroy/building-faster-accurate-cord-search-engine)

***

