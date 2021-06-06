# simpleT5 - Train T5 models in 3 lines of Code with ONNX Inference

[**simpleT5**](https://github.com/Shivanandroy/simpleT5) is built on top of PyTorch-lightning⚡️ and Transformers🤗 that lets you quickly train/fine-tune T5 models along with ONNX inference support

![](https://miro.medium.com/max/2416/1*WqDq6TFR3ETjZSeb_tbc0A.png)

With simpleT5 — It is super easy to fine-tune any T5 model on your dataset (Pandas dataframe )— for any task (summarization, translation, question-answering, or other sequence-to-sequence tasks), 
just — **import, instantiate, download a pre-trained model and train**.

Checkout the [**simpleT5 GitHub repository**](https://github.com/Shivanandroy/simpleT5) (drop a 🌟 if you like it) along with example [**Colab notebook**](https://colab.research.google.com/drive/1JZ8v9L0w0Ai3WbibTeuvYlytn0uHMP6O?usp=sharing).

![](https://miro.medium.com/max/875/1*8u3wdbJWwwG0qmvHZDnc2A.png)

Before we jump on how to use simpleT5, a quick introduction about T5 —

## What is T5 ?
A `T5` is an encoder-decoder model. It converts all NLP problems like language translation, summarization, text generation, question-answering, to a text-to-text task. 


For e.g., in case of **translation**, T5 accepts `source text`: English, as input and tries to convert it into `target text`: Serbian: 
| source text 	| target text 	|
|-	|-	|
| Hey, there! 	| Хеј тамо! 	|
| I'm going to train a T5 model with PyTorch 	| Обучићу модел Т5 са ПиТорцх-ом 	|


In case of **summarization**, source text or input can be a long description and target text can just be a one line summary. 
| source text                                                                                                                                                                                                                                                                                                                                                                               	| target text                                                       	|
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|-------------------------------------------------------------------	|
| "Saurav Kant, an alumnus of upGrad and IIIT-B's PG Program in Machine learning and Artificial Intelligence, was a Sr Systems Engineer at Infosys with almost 5 years of work experience. The program and upGrad's 360-degree career support helped him transition to a Data Scientist at Tech Mahindra with 90% salary hike. upGrad's Online Power Learning has powered 3 lakh+ careers." 	| upGrad learner switches to career in ML & Al with 90% salary hike 	|


Now, let’s fine-tune a T5 model on summarization task with simpleT5 —

## Data
We will use a news-summary dataset, for summarization. This dataset has 2 columns: text — which has the actual news and headlines — is one line summary of the news.

```python
# let's import a dataset
>> import pandas as pd
>> from sklearn.model_selection import train_test_split

>> path = "https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv"
>> df = pd.read_csv(path)
>> df.head()
```
```
>> output:
  
 	headlines 	                                          text
0 	upGrad learner switches to career in ML & Al w... 	Saurav Kant, an alumnus of upGrad and IIIT-B's...
1 	Delhi techie wins free food from Swiggy for on... 	Kunal Shah's credit card bill payment platform...
2 	New Zealand end Rohit Sharma-led India's 12-ma... 	New Zealand defeated India by 8 wickets in the...
3 	Aegon life iTerm insurance plan helps customer... 	With Aegon Life iTerm Insurance plan, customer...
4 	Have known Hirani for yrs, what if MeToo claim... 	Speaking about the sexual harassment allegatio...
```

simpleT5 expects a pandas dataframe with 2 columns — **source_text** and **target_text**. As we are summarizing news articles, we want our T5 model to learn how to convert **actual news (text column) → one line summary (headlines column)**. So, our source_text will be the text column, and target_text will be the headlines column.

T5 also expects a task-related prefix — to uniquely identify the task that we want to perform on our dataset. Let’s add “summarize: “ as a prefix to our source_text.

```python
# simpleT5 expects dataframe to have 2 columns: "source_text" and "target_text"
df = df.rename(columns={"headlines":"target_text", "text":"source_text"})
df = df[['source_text', 'target_text']]

# T5 model expects a task related prefix: since it is a summarization task, we will add a prefix "summarize: "
df['source_text'] = "summarize: " + df['source_text']
print(df)
```
```

>> output: 
 	                    source_text 	                                    target_text
0 	summarize: Saurav Kant, an alumnus of upGrad a... 	upGrad learner switches to career in ML & Al w...
1 	summarize: Kunal Shah's credit card bill payme... 	Delhi techie wins free food from Swiggy for on...
2 	summarize: New Zealand defeated India by 8 wic... 	New Zealand end Rohit Sharma-led India's 12-ma...
3 	summarize: With Aegon Life iTerm Insurance pla... 	Aegon life iTerm insurance plan helps customer...
4 	summarize: Speaking about the sexual harassmen... 	Have known Hirani for yrs, what if MeToo claim...
```

Next, let’s split our dataset into train and test — and we’re done.
```python
train_df, test_df = train_test_split(df, test_size=0.2)
train_df.shape, test_df.shape
```
```
>> output:
  ((78720, 2), (19681, 2))
```

## Train
We will import SimpleT5 class, download a pre-trained T5 model and then train it on our dataset — train_df and test_df. You can also specify other optional model arguments, such as — **source_max_token_len, target_max_token_len, batch_size, epochs, early_stopping** etc.

```python
from simplet5 import SimpleT5

model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")
model.train(train_df=train_df,
            eval_df=test_df, 
            source_max_token_len=128, 
            target_max_token_len=50, 
            batch_size=8, 
            max_epochs=3, 
            use_gpu=True
           )
```
And Voila — You’re done !🎉

## Inference
For inference— you can load a trained model, and use **predict** function to generate outputs

```python
# let's load the trained model for inferencing:
model.load_model("t5","outputs/SimpleT5-epoch-2-train-loss-0.9526", use_gpu=True)

text_to_summarize="""summarize: Rahul Gandhi has replied to Goa CM Manohar Parrikar's letter, 
which accused the Congress President of using his "visit to an ailing man for political gains". 
"He's under immense pressure from the PM after our meeting and needs to demonstrate his loyalty by attacking me," 
Gandhi wrote in his letter. Parrikar had clarified he didn't discuss Rafale deal with Rahul.
"""
model.predict(text_to_summarize)

>> output:
  
  ["Rahul responds to Parrikar's letter accusing him of visiting ailing man"]
```
## Quantization and ONNX support
For faster inference on CPU up to 3X, you can quantize your trained model and use ONNX model to generate outputs with **onnx_predict**

```python
# for faster inference on cpu, quantization, onnx support:
>> model.convert_and_load_onnx_model(model_dir="outputs/SimpleT5-epoch-2-train-loss-0.9526")

  exporting to onnx... |################################| 3/3
  Quantizing... |################################| 3/3
  
```
```  
>> model.onnx_predict(text_to_summarize)

CPU times: user 754 ms, sys: 23.5 ms, total: 777 ms
Wall time: 799 ms

[Rahul responds to Parrikar's letter accusing him of visiting Goa]
```

The complete script looks like below —

```python
# !pip install simplet5

# --> Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

path = "https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv"
df = pd.read_csv(path)


# --> preprocessing dataset: training_df, test_df with "source_text" & "target_text" columns

# simpleT5 expects dataframe to have 2 columns: "source_text" and "target_text"
df = df.rename(columns={"headlines":"target_text", "text":"source_text"})
df = df[['source_text', 'target_text']]

# T5 model expects a task related prefix: since it is a summarization task, we will add a prefix "summarize: "
df['source_text'] = "summarize: " + df['source_text']

train_df, test_df = train_test_split(df, test_size=0.2)


# --> Finetuning T5 model with simpleT5

from simplet5 import SimpleT5

model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")
model.train(train_df=train_df,
            eval_df=test_df, 
            source_max_token_len=128, 
            target_max_token_len=50, 
            batch_size=8, max_epochs=3, use_gpu=True)


# --> Load and inference

# let's load the trained model for inferencing:
model.load_model("t5","outputs/SimpleT5-epoch-2-train-loss-0.9526", use_gpu=True)

text_to_summarize="""summarize: Rahul Gandhi has replied to Goa CM Manohar Parrikar's letter, 
which accused the Congress President of using his "visit to an ailing man for political gains". 
"He's under immense pressure from the PM after our meeting and needs to demonstrate his loyalty by attacking me," 
Gandhi wrote in his letter. Parrikar had clarified he didn't discuss Rafale deal with Rahul.
"""
model.predict(text_to_summarize)

# --> model quantization & ONNX support

# for faster inference on cpu, quantization, onnx support:
model.convert_and_load_onnx_model(model_dir="outputs/SimpleT5-epoch-2-train-loss-0.9526")
model.onnx_predict(text_to_summarize)
```
