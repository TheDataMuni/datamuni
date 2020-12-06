### Disclaimer: This article is not an introduction to Apache Kafka and expects the reader to have some basic understanding of the various terminologies related to Kafka.

### Implementing an Apache Kafka Producer in Python for realtime BBC News Feed
The following article will help you setup Kafka in your local machine and let you read data from BBC RSS feeds and push it to a Kafka topic.

#### Setting up Apache Kafka and Zookeeper in your local machine
Inorder to follow the tutorial, one needs to install Apache Kafka and Apache Zookeeper in their local machine. Also one must have JAVA installed in the system. If you are a MAC user, you can install the above using
```
$ brew cask install java
$ brew install kafka

```
Once the above two requirements are satisfied, lets start the zookeeper and kafka by running the following commands in two seperate terminals.

To start zookeeper:
```
zookeeper-server-start /usr/local/etc/kafka/zookeeper.properties

```
The zookeeper will start at the port 121.0.0.1:2181

To start kafka broker
```
kafka-server-start /usr/local/etc/kafka/server.properties

```
The kafka broker will be at 212.0.0.1:9092

Now, lets create a topic within the broker. Inorder to do that, 
1. Open a new terminal
2. Run the following command
```
kafka-topics --zookeeper 127.0.0.1:2181 --topic bbcfeed --create --partitions 3 --replication-factor 1
```
The above command will create a topic `bbcfeed` with `3` partitions. Since we are starting a single broker, the replication-factor should always be 1. As a general rule, the `--replication-factor` should be less than or equal to the number of brokers in the cluster. If the topic is successfully created, run the `--describe` function to know more details about the topic. 
```
kafka-topics --zookeeper 127.0.0.1:2181 --topic bbcfeed --describe

```
Output
```
Topic: bbcfeed	PartitionCount: 3	ReplicationFactor: 1	Configs: 
	Topic: bbcfeed	Partition: 0	Leader: 0	Replicas: 0	Isr: 0
	Topic: bbcfeed	Partition: 1	Leader: 0	Replicas: 0	Isr: 0
	Topic: bbcfeed	Partition: 2	Leader: 0	Replicas: 0	Isr: 0
```
Since we arent implementing a consumer in this tutorial but if you want to see the output from the producer we are going to implement, you can create a kafka-consumer from the CLI. To do that run,
```
kafka-console-consumer --bootstrap-server 127.0.0.1:9092 --topic bbcfeed
```

#### Requirements
To create a kafka producer in python, one must have `kafka-python` package installed. There are few other packages required which can be installed using pip
```
pip install kafka-python
pip install beautifulsoup4
pip install pandas
```
Once the above packages are successfully installed, we can go to the code. For the ease of development, there will be two files
1. bbc_xml_feed_parser.py -  This will help us parsing data from the bbc RSS feed http://feeds.bbci.co.uk/news/world/rss.xml. The data from the feed will be extracted using beautifulsoup and some transformations will be done on it such as ordering the news articles based on their date of publishing.

```
from bs4 import BeautifulSoup
import requests
import pandas as pd

class BBCParser():
    """
    Class to read from BBC RSS feed
    """
    
    def __init__(self):
        self.bbc_url = "http://feeds.bbci.co.uk/news/world/rss.xml"
        self.response = None
        self.status = 404  
        self.items=[]
        
    def getResponse(self):
        """
        Function to read from BBC RSS Feed

        Returns
        -------
        TYPE: Integer
            Status code, 200 if success else 404
        TYPE: ResultSet
            Response from BBC RSS feed

        """
        
        self.response = requests.get(self.bbc_url)
        self.response = BeautifulSoup(self.response.content, features= 'xml')
        
        if (self.response !=None):
            if(self.response.find_all('link')[0].text == 'https://www.bbc.co.uk/news/'):  
                self.status = 200
                self.items = self.response.find_all('item')
        return self.status, self.items
        
    
    def responseParser(self, items):
        """
        Function to parse the feed and get elements required from it.

        Parameters
        ----------
        items : List
            List of all items parsed from the XML Feed

        Returns
        -------
        TYPE: List
            List of interested items parsed from the XML Feed
        TYPE: String
            Top item from the parsed XML Feed

        """
        parsedItems=[]
        for item in items:
            item_dict = {}
            item_dict['title'] = item.title.text
            item_dict['link'] = item.link.text
            item_dict['createdOn'] = item.pubDate.text
            parsedItems.append(item_dict)    
        return parsedItems
    
    def newsOrganiser(self, parser_output):
        """
        Function to reorder dataframe based on timestamp

        Parameters
        ----------
        parser_output : Dataframe
            Pandas df output from responseParser method.

        Returns
        -------
        final_news_dict : List of dicts
            Dictionary of reordered records.
        top_news : string
            Top element from title field.

        """
        news_df = pd.DataFrame(parser_output)
        news_df['TS'] = news_df['createdOn'].apply(lambda x:pd.Timestamp(x))
        news_df['PublishDateTime'] = pd.to_datetime(news_df['TS'], format='%Y-%m-%d %H:%M:%S-%Z',errors='coerce').astype(str)
        news_df = news_df.sort_values('PublishDateTime', ascending=False, ignore_index=True)
        final_news_df = news_df.drop(['createdOn','TS'], axis=1)
        top_news = final_news_df['title'].iloc[0]
        final_news_dict = final_news_df.to_dict(orient='records')
        return final_news_dict, top_news
```

2. kafka_producer.py - This is the main script which will publish the data to the concerned kafka topic, `bbcfeed`. The producer will publish data to the topic every one hour if and only if there is a new article in the RSS stream.
```
from bbc_xml_feed_parser import BBCParser
from kafka import KafkaProducer
import time
import json


def json_serializer(payload):
    """
    

    Parameters
    ----------
    payload : Dict
        Dictionary of data values that needs to be serialized before sending to Kafka topics.

    Returns
    -------
    return_payload : json
        json data encoded to utf-8
        
    """
    return_payload = json.dumps(payload).encode('utf-8')
    return return_payload


if __name__=='__main__':
    bbc = BBCParser()
    prev_top_news = None
    top_news = None
    
    bootstrap_servers = '127.0.0.1:9092'
    client_id = 'bbc_feed_publisher'
    topic = 'bbcfeed'
    retries = 5
    
    producer = KafkaProducer(bootstrap_servers = bootstrap_servers, client_id = client_id, retries = retries)
    
    while(True):
        status_code, items = bbc.getResponse()
        if(status_code == 200):
            parser_output = bbc.responseParser(items)
            final_news, top_news = bbc.newsOrganiser(parser_output)
            
        #Condition to check if its necessary to publish message to Kafka or not
        if(top_news == prev_top_news):
            print("Do not publish to Kafka")
        else:
            for news in final_news:
                if(news!= prev_top_news):        
                    print("Publishing to Kafka")
                    producer.send(topic, value=json_serializer(news))
                    time.sleep(0.5)
                else:
                    break
            prev_top_news = top_news
        print('Producer Disabled for 1 hr')
        time.sleep(3600)
```

The code above is pretty much self explanatory, but in case you need further clarifications reach out at rohitanil93@gmail.com. If the PR gets approved, I will soon write an article on implementing a Kafka Consumer which will read news from the broker and classify the news feed into various categories using machine learning.
