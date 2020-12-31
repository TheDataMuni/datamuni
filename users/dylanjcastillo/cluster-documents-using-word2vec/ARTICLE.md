# Cluster Documents Using Word2Vec
 
## Table of Contents

- [Intro](#intro)
- [How to Cluster Documents](#how-to-cluster-documents)
- [Sample Project: Clustering News Articles](#sample-project--clustering-news-articles)
  - [Set Up Your Local Environment](#set-up-your-local-environment)
  - [Import the Required Libraries](#import-the-required-libraries)
  - [Clean and Tokenize Data](#clean-and-tokenize-data)
  - [Generate Document Vectors](#generate-document-vectors)
    - [Train Word2Vec Model](#train-word2vec-model)
    - [Create Document Vectors from Word Embedding](#create-document-vectors-from-word-embedding)
  - [Cluster Documents Using (Mini-batches) K-means](#cluster-documents-using--mini-batches--k-means)
    - [Creation of Clusters](#creation-of-clusters)
    - [Qualitative Review of Clusters](#qualitative-review-of-clusters)
- [Conclusion](#conclusion)

## Intro

I bet you can find more sentiment analysis tutorials online than people doing sentiment analysis in their day jobs. Don't get me wrong. I'm not saying those tutorials aren't useful. I just want to highlight that **supervised learning** receives a lot more attention than any other method in Natural Language Processing (NLP).

Oddly enough, there's a big chance that most of the text data you'll use in your next projects won't have ground truth labels. So supervised learning might not be a solution you can apply to your data problems right away.

What can you do then? Use **unsupervised learning** algorithms.

In this tutorial, you'll learn to apply unsupervised learning to generate value from your text data. You'll cluster documents using a mix of a word embedding (Word2Vec) and the K-means algorithm.

The tutorial focuses on practical manners. You won't find much theory in the next sections, aside from brief definitions of relevant ideas.

To make the most of this tutorial, you should be familiar with these topics:

- [Supervised and unsupervised learning](https://datamuni.com/@abhishek/supervised-vs-unsupervised-learning)
- [Clustering](https://en.wikipedia.org/wiki/Cluster_analysis) (particularly, [K-means](https://realpython.com/k-means-clustering-python/))
- [Word2Vec](https://jalammar.github.io/illustrated-word2vec/)

Let's get to it!

## How to Cluster Documents

You can think of the process of clustering documents in three steps:

1. [**Clean and tokenize data:**](https://dylancastillo.co/nlp-snippets-clean-and-tokenize-text-with-python/) this usually involves lowercasing text, removing non-alphanumeric characters, or stemming words.
2. **Generate vector representations of the documents**: this concerns the mapping of documents from words into numerical vectors—some common ways of doing this include using bag-of-words models or word embeddings.
3. **Apply clustering algorithm on the document vectors**: this requires selecting and applying a clustering algorithm to find the best possible groups using the document vectors. Some frequently used algorithms include K-means, DBSCAN, or Hierarchical Clustering.

That's it! Now, you'll see how that looks in practice.

## Sample Project: Clustering News Articles

In this section, you'll learn how to cluster documents by working through a small project. You'll group news articles into categories using a [dataset](https://www.kaggle.com/szymonjanowski/internet-articles-data-with-users-engagement) published by [Szymon Janowski](https://github.com/sleter).

### Set Up Your Local Environment

To follow along with the tutorial examples, you'll need to download the data and install a few libraries. You can do it by following these steps:

1. Clone the [nlp-snippets repository](https://github.com/dylanjcastillo/nlp-snippets/) locally.
2. Create a new virtual environment using `venv` or `conda`.
3. Activate your new virtual environment.
4. Install the required libraries.
5. Start a Jupyter notebook.

If you're using `venv`, then you need to run these commands:

```shell
$ git clone XXX
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements
(venv) $ jupyter notebook
```

If you're using `conda`, then you need to run these commands:

```shell
$ git clone XXX
$ conda create --name venv
$ conda activate venv
(venv) $ pip install -r requirements
(venv) $ jupyter notebook
```

Next, open the Jupyter notebook. Then, create a new notebook in the root folder and set its name to `clustering_word2vec.ipynb`.

By now, your project structure should look like this:

```text
nlp-snippets/
│
├── clustering/
│
├── data/
│
├── ds_utils/
│
├── preprocessing/
│
├── venv/ # (If you're using venv)
│
├── clustering_word2vec.ipynb
├── LICENSE
├── README.md
└── requirements.txt
```

This is your project's structure. It includes these directories and files:

- `clustering/`: Examples of clustering text data using bag-of-words, training a word2vec model, and using a pretrained fastText embeddings.
- `data/`: Data used for the clustering examples.
- `ds_utils/`: Common utility functions used in the sample notebooks in the repository.
- `preprocessing/`: Code snippets frequently used for preprocessing text.
- `venv/`: If you used `venv`, then this directory will contain the files related to your virtual environment.
- `requirements.txt`: Libraries used in the examples provided.
- `README` and `License`: Information about the repository, and its license.

For now, you'll use the notebook you created (`clustering_word2vec.ipynb`) and the news dataset in `data/`. The notebooks in `clustering/` and `preprocessing/` include additional code snippets that might be useful for NLP tasks. You can review those on your own.

In the next section, you'll create the whole pipeline from scratch. If you'd like to download the full — and cleaner — version of the code in the examples, then go to the [NLP Snippets repository](https://github.com/dylanjcastillo/nlp-snippets/blob/main/clustering/word2vec.ipynb).

That's it for set up! Next, you'll define your imports.

### Import the Required Libraries

Once you finish setting up your local environment, it's time to start writing code in your notebook. Open `clustering_word2vec.ipynb`, and copy the following code in the first cell:

```python
import os
import random
import re
import string

import nltk
import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score

nltk.download("stopwords")

SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
```

These are the libraries you need for the sample project. Here's what you do with each of them:

- `**os**` and `**random**` help you define a random seed to make the code deterministically reproducible.
- `**re**` and `**string**` provide you with easy ways to clean the data.
- `**pandas**`helps you read the data.
- `**numpy**`provides you with linear algebra utilities you'll use to evaluate results. Also, it's used for setting a random seed to make the code deterministically reproducible.
- `**gensim**` makes it easy for you to train a word embedding from scratch using the `Word2Vec` class.
- `**nltk**`aids you cleaning and tokenizing data through the use of the `word_tokenize` method and the `stopword` list.
- `**sklearn**`gives you an easy interface to the clustering model, `MiniBatchKMeans`, and the metrics to evaluate the quality of its results, `silhouette_samples` and `silhouette_score`.

In addition to importing the libraries, you download English stopwords using `nltk.download("stopwords")`, you define `SEED` and set it as random seed using `numpy`, `random`, and the `PYTHONHASHSEED` environment variable. This last step makes sure your code is reproducible across systems.

Run this cell and make sure you don't get any errors. In the next section, you'll prepare your text data.

### Clean and Tokenize Data

After you import the required libraries, you need to read and preprocess the data you'll use in your clustering algorithm. The preprocessing consists of cleaning and tokenizing the data. To do that, copy the following function in a new cell in your notebook:

```python
def clean_text(text, tokenizer, stopwords):
    """Pre-process text and generate tokens

    Args:
        text: Text to tokenize.

    Returns:
        Tokenized text.
    """
    text = str(text).lower()  # Lowercase words
    text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = re.sub(
        f"[{re.escape(string.punctuation)}]", "", text
    )  # Remove punctuation

    tokens = tokenizer(text)  # Get tokens from text
    tokens = [t for t in tokens if not t in stopwords]  # Remove stopwords
    tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
    tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
    return tokens
```

This code cleans and tokenizes a `text` input, using a predefined `tokenizer` and a list of `stopwords`. It performs the following operations on the text provided:

1. Transforms the input into a string and lowercases it
1. Removes substrings like "[+300 chars]" that I found while reviewing the data
1. Removes multiple spaces, tabs, and line breaks
1. Removes ellipsis characters
1. Replaces dash between words with space and removes punctuation
1. Tokenizes text and removes tokens using a list of stop words
1. Removes digits and tokens which length is too short

Then, in the next cell, copy the following code to read the data and apply that function to the text columns:

```python
custom_stopwords = set(stopwords.words("english") + ["news", "new", "top"])
text_columns = ["title", "description", "content"]

df = df_raw.copy()
df["content"] = df["content"].fillna("")

for col in text_columns:
    df[col] = df[col].astype(str)

# Create text column based on title, description, and content
df["text"] = df[text_columns].apply(lambda x: " | ".join(x), axis=1)
df["tokens"] = df["text"].map(lambda x: clean_text(x, word_tokenize, custom_stopwords))

# Remove duplicated after preprocessing
_, idx = np.unique(df["tokens"], return_index=True)
df = df.iloc[idx, :]

# Remove empty values and keep relevant columns
df = df.loc[df.tokens.map(lambda x: len(x) > 0), ["text", "tokens"]]

print(f"Original dataframe: {df_raw.shape}")
print(f"Pre-processed dataframe: {df.shape}")
```

This is how you read and preprocess the data. This code applies the cleaning function you defined earlier, removes duplicates and nulls, and drops irrelevant columns.

You apply these steps to a new dataframe (`df`) you'll use for the next examples. It contains a column with the raw documents called `text` and another one with the preprocessed documents called `tokens`.

If you execute the two cells you defined, then you should get the following output:

```text
Original dataframe: (10437, 15)
Pre-processed dataframe: (9882, 2)
```

Next, you'll create document vectors using Word2Vec.

### Generate Document Vectors

After you've cleaned and tokenized the text, you'll use the documents' tokens to create vectors using Word2Vec. This process consists of two steps:

1. Train a Word2Vec model using your tokenized documents.
2. Generate a vector per document based on its individual word vectors.

In this section, you'll go through these steps.

#### Train Word2Vec Model

The following code will help you train a Word2Vec model. Copy it into a new cell in your Jupyter notebook:

```python
model = Word2Vec(sentences=tokenized_docs, vector_size=100, workers=1, seed=SEED)
```

You use this code to train a Word2Vec model based on your tokenized documents. For this example, you specified the following parameters in the `Word2Vec` class:

- `sentences` expects a list of lists with the tokenized documents.
- `vector_size` defines the size of the word vectors. In this case, you set it to 100.
- `workers` defines how many cores you use for training. I set it to 1 to make sure the code is determinastically reproducible.
- `seed` seed for random number generator. It's set to the constant `SEED` you defined in the first cell.

There's other parameters you can tune when training the Word2Vec model. Take a look at [gensim's documentation](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec) if you'd like to learn more about them.

Now, run the cell in the Jupyter notebook. It might take a couple of minutes. After its done, you can validate that the results make sense by [plotting the vectors](https://machinelearningmastery.com/develop-word-embeddings-python-gensim/) or by reviewing the similarity results for relevant words. You can do the latter by copying and running this code in a cell in your notebook:

```python
model.wv.most_similar("trump")
```

If you run this code, then you'll get this output:

```text
[('trumps', 0.988541841506958),
    ('president', 0.9746493697166443),
    ('donald', 0.9274922013282776),
    ('ivanka', 0.9203903079032898),
    ('impeachment', 0.9195784330368042),
    ('pences', 0.9152231812477112),
    ('avlon', 0.9148306846618652),
    ('biden', 0.9146010279655457),
    ('breitbart', 0.9144087433815002),
    ('vice', 0.9067237973213196)]
```

That's it! You've trained your Word2Vec model, now you'll use it to generate document vectors.

#### Create Document Vectors from Word Embedding

Now you'll generate document vectors using the Word2Vec model you trained. The idea is straightforward. From the Word2Vec model you'll get numerical vectors per each of the words in a document, so you need to find a way of generating a single vector out of them. A common approach is to use the [average](https://stats.stackexchange.com/a/318891) of the vectors.

The following code will help you create a vector per document by averaging its word vectors. Create a new cell in your Jupyter notebook and copy this code there:

```python
def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features

vectorized_docs = vectorize(tokenized_docs, model=model)
len(vectorized_docs), len(vectorized_docs[0])
```

This code will get all the word vectors of each document and average them to generate a vector per each document. Here's what's happening there:

1. You define the `vectorize` function that takes a list of documents and a `gensim` model and generates a feature vector per document.
2. You apply the function to the tokenized documents in `tokenized_doc`, using the Word2Vec `model` you trained earlier.
3. You print the length of the list of documents and the size of the generated vectors.

Next, you'll cluster the documents using Mini-batches K-means.

### Cluster Documents Using (Mini-batches) K-means

To cluster the documents, you'll use the **Mini-batches K-means **algorithm. This variant of K-means uses random samples of the input data to reduce the time required during training. The upside is that it shares the same objective function with the original algorithm so, in practice, the results are [[j](https://scikit-learn.org/stable/modules/clustering.html#mini-batch-k-means)ust a bit worse than K-means](https://scikit-learn.org/stable/modules/clustering.html#mini-batch-k-means).

In the code snippet below, you can see the function you'll use to create the clusters using Mini-batches K-means. Create a new cell in your notebook, and copy the following code there:

```python
    def mbkmeans_clusters(
    	X,
        k,
        mb,
        print_silhouette_values,
    ):
        """Generate clusters and print Silhouette metrics using MBKmeans

        Args:
            X: Matrix of features.
            k: Number of clusters.
            mb: Size of mini-batches.
            print_silhouette_values: Print silhouette values per cluster.

        Returns:
            Trained clustering model and labels based on X.
        """
        km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
        print(f"For n_clusters = {k}")
        print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
        print(f"Inertia:{km.inertia_}")

        if print_silhouette_values:
            sample_silhouette_values = silhouette_samples(X, km.labels_)
            print(f"Silhouette values:")
            silhouette_values = []
            for i in range(k):
                cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
                silhouette_values.append(
                    (
                        i,
                        cluster_silhouette_values.shape[0],
                        cluster_silhouette_values.mean(),
                        cluster_silhouette_values.min(),
                        cluster_silhouette_values.max(),
                    )
                )
            silhouette_values = sorted(
                silhouette_values, key=lambda tup: tup[2], reverse=True
            )
            for s in silhouette_values:
                print(
                    f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
                )
        return km, km.labels_
```

This function creates the clusters using the Mini-batches K-means algorithm. It takes the following arguments:

- `**X**`: Matrix of features. In this case, it's your vectorized documents.
- `**k**`:Number of clusters you'd like to create.
- `**mb**`: Size of mini-batches.
- `**print_silhouette_values**`: Defines if the Silhouette Coefficient is printed for each cluster. If you haven't heard about this coefficient, don't worry, you'll learn about it in a bit!

`mbkmeans_cluster` takes these arguments and returns the fitted clustering model and the labels for each document.

Run the cell where you copied the function. Next, you'll apply this function to your vectorized documents.

#### Creation of Clusters

Now, you need to execute `mbkmean_clusters` providing it with the vectorized documents, and the number of clusters. You'll print the Silhouette Coefficients per cluster, as that will help you review the quality of your clusters.

Create a new cell and copy this code there:

```python
clustering, cluster_labels = mbkmeans_clusters(
    X=vectorized_docs,
    k=50,
    mb=500,
    print_silhouette_values=True,
)
df_clusters = pd.DataFrame({
    "text": docs,
    "tokens": [" ".join(text) for text in tokenized_docs],
    "cluster": cluster_labels
})
```

This code will fit the clustering model, print the Silhouette Coefficient per cluster, and return the fitted model and the labels per cluster. It'll also create a dataframe you can use to review the results.

There are a few things to consider when setting the input arguments:

- `print_silhouette_values` is straightforward. In this case, you set it to `True` to print the evaluation metric per cluster. This will help you review the results.
- `mb` depends on the size of your dataset. You need to make sure that it is not too small to avoid a significant impact on the quality of results and not too big to avoid making the execution too slow. In this case, you set it to 500 observations.
- `k` is trickier. In general, it involves a mix of qualitative analysis and quantitative metrics. After a few experiments on my side, I found that 50 seemed to work well. But that is more or less arbitrary.

For the quantitative evaluation of the number of clusters, you could use metrics like the **Silhouette Coefficient**. This coefficient is an evaluation metric frequently used in problems where ground truth labels are not known. It's calculated using the mean intra-cluster distance and the mean nearest-cluster distance and goes from -1 to 1. Well defined clusters result in positive values of this coefficient, while incorrect clusters will result in negative values. If you'd like to learn more about it, then look at [scikit-learn's documentation](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient).

The qualitative part generally requires you to have domain knowledge of the subject matter so that you can sense-check the results of your clustering algorithm. In the next section, I'll show you two approaches you can use to do a qualitative check of your results.

After executing the cell you just created, the output should look like this:

```text
For n_clusters = 50
Silhouette coefficient: 0.11
Inertia:3568.342791047967
Silhouette values:
    Cluster 29: Size:50 | Avg:0.39 | Min:0.01 | Max: 0.59
    Cluster 35: Size:30 | Avg:0.34 | Min:0.05 | Max: 0.54
    Cluster 37: Size:58 | Avg:0.32 | Min:0.09 | Max: 0.51
    Cluster 39: Size:81 | Avg:0.31 | Min:-0.05 | Max: 0.52
    Cluster 27: Size:63 | Avg:0.28 | Min:0.02 | Max: 0.46
    Cluster 6: Size:101 | Avg:0.27 | Min:0.02 | Max: 0.46
    Cluster 24: Size:120 | Avg:0.26 | Min:-0.04 | Max: 0.46
    Cluster 49: Size:65 | Avg:0.26 | Min:-0.03 | Max: 0.47
    Cluster 47: Size:53 | Avg:0.23 | Min:0.01 | Max: 0.45
    Cluster 22: Size:78 | Avg:0.22 | Min:-0.01 | Max: 0.43
    Cluster 45: Size:38 | Avg:0.21 | Min:-0.07 | Max: 0.41
...
```

This is the output of your clustering algorithm. The sizes and Silhouette Coefficients per cluster are the most relevant metrics. The clusters are printed by the value of the Silhouette coefficient in descending order. A higher score means denser – and thus better – clusters. In this case, you can see that clusters 29, 35, and 37 seem to be the top ones.

Next, you'll learn how to check what's in each cluster.

### Qualitative Review of Clusters

There are a few ways you can do a qualitative analysis of the results. During the earlier sections, the approach you took results in vector representations of tokens and documents and vectors of the clusters' centroids. To analyze the results, you can find the most representative tokens and documents by looking for the vectors closest to the clusters' centroids.

Here's how you obtain the most representative tokens per cluster:

```python
print("Most representative terms per cluster (based on centroids):")
for i in range(50):
    tokens_per_cluster = ""
    most_representative = model.wv.most_similar(positive=[clustering.cluster_centers_[i]], topn=5)
    for t in most_representative:
        tokens_per_cluster += f"{t[0]} "
    print(f"Cluster {i}: {tokens_per_cluster}")
```

For the top clusters we identified earlier – 29, 35, and 37 – these are the results:

```text
Cluster 29: noaa sharpie claim assertions forecasters
Cluster 35: eye lilinow path halts projected
Cluster 37: cnnpolitics complaint clinton pences whistleblower
```

Next, we can do the same analysis but with documents instead of tokens. This is how you find the most representative documents for cluster 29:

```python
test_cluster = 29
most_representative_docs = np.argsort(
    np.linalg.norm(vectorized_docs - clustering.cluster_centers_[test_cluster], axis=1)
)
for d in most_representative_docs[:3]:
    print(docs[d])
    print("-------------")
```

And these are the 3 most representative documents in that cluster:

```text
Dorian, Comey and Debra Messing: What Trump tweeted on Labor Day weekend | President Donald Trump axed his visit to Poland over the weekend to monitor Hurricane Dorian from Camp David with emergency management staff, but if the President's more than 120 tweets are any indication, he had more than just the storm on his mind. | Washington (CNN)President Donald Trump axed his visit to Poland over the weekend to monitor Hurricane Dorian from Camp David with emergency management staff, but if the President's more than 120 tweets are any indication, he had more than just the storm on hi… [+3027 chars]
-------------
Ross Must Resign If Report He Threatened NOAA Officials Is True: Democrat | As President Donald Trump claimed Hurricane Dorian could hit Alabama, the National Weather Service tweeted to correct the rumors. | Commerce Secretary Wilbur Ross is facing calls to resign over a report alleging that he threatened to fire top officials at NOAA for a tweet disputing President Donald Trump's claim that Hurricane Dorian would hit Alabama.
"If that story is true, and I don't… [+3828 chars]
-------------
Federal weather workers are furious at the NOAA's 'utterly disgusting' statement defending Trump's claim Hurricane Dorian would hit Alabama | Federal weather workers have reacted furiously to the National Oceanic and Atmospheric Administration's (NOAA) defence of US President Donald Trump's repeated assertions that Hurricane Dorian was set to hit Alabama. "Never ever before has their management thr… | Federal weather workers have reacted furiously to the National Oceanic and Atmospheric Administration's (NOAA) defence of US President Donald Trump's repeated assertions that Hurricane Dorian was set to hit Alabama, saying they have been "thrown under the bus… [+3510 chars]
```

Most of the results seem to be related to a dispute between Donald Trump and the National Oceanic and Atmospheric Agency (NOAA). It was a fairly popular controversy that ended up being called the [Sharpiegate](https://en.wikipedia.org/wiki/Hurricane_Dorian%E2%80%93Alabama_controversy).

You could also explore other approaches like generating word frequencies per cluster or review random samples of documents per cluster.

## Conclusion

Way to go! You just learned how to cluster documents using Word2Vec. There are other approaches you could take to cluster text data like:

- Using a pre-trained word embedding instead of training your own.
- Generating feature vectors using a bag-of-words approach instead of word embeddings.
- Reducing dimensionality of feature vectors. This is very useful if you use a bag-of-words approach.
- Clustering documents using other algorithms like HDBSCAN, or Hierarchical Clustering.

I hope you find this tutorial useful. Shoot me a message if you have any questions!
