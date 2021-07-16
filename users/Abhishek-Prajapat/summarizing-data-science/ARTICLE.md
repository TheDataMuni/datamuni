# Summarizing Data Science

Hi! This is my first article and I want to share what I have learned so far in Data-Science in a broad prospective. And this is specially for those who have started this field and has worked in for about a week.

#### Sorry if there are any spelling mistakes.

### Note- be ready to google things to gain most of this.


## What are the Prerequisites?

Yeah you all know that Probability, Stats, calculas are needed (not in depth though) but I believe there are more things going behind.
Wherever you are, either in Data-Science, ML or... you know what just take the whole **Computer-Science** field there are certains thing you need to know which are not compulsory while learning cause you know what everyone assumes you know this. These are especially necessary when you work.

 - Version-control ( This helps everywhere in all domains )
 - Basics of your OS/Terminal ( It will make you less frustrated )
 - Object Oriented Programming ( Is covered in Python courses but never focused over )
 - Basics of SQL ( again helps everywhere )
 - Start reading your errors carefully and remember them, this will improve your google query.

Once again these are not compulsory but it won't take much to get the hang of these and if you these your  journey will be a lot easier.

## DS OR Data-Science

**This field has fake hype**

This is by far the largest part and you need to do a lot of things. For this you need to know the following things :-

 - **Big-Data-Analysis**
	 - I not sure myself but I have seen it in the requirements.
 - **How to train ML/DL models**
 - **How to deploy models**
 - **How to maintain models**
 - **Yup! present to your Boss**
 - **Did I forget to mention that the data is continuosly coming?**
 - **Sometimes you need to collect data by yourself**

You don't need to be expert on all these thing and your data will also be somthing which you possible have worked upon in some way.
As you can see its a lot of work for a single guy and hence is mostly found in startups. 

**So there comes the question what is the thing for Big-Companies?**
Well in Big-Companies all those above mention tasks are divided among several posts.
- Big-Data Engineer
- Data-Analyst
- Data-Scientist
	- Mostly they only need to train models using optimmised algorithms produced by DS-Core and make that model production ready and maintain it.
- Data-Science Core
	- After you have taken a look at this field and may be around 6month then you will come to know that Giant companies have Giant resources and Giant Data ( After all they are Giant ). That also means that they need to process that data more fast and efficent which leads them to be more specific to their data and develop entirely new methods.


**As you have seen that this field need a lot and almost no-one talking about data science talks anything other than training ML models this field has fake hype. Which lead to where does that hype come from? ( Mostly Salary and Deep Learning )**


## ML OR Machine-Learning

Yeah it also includes the sub-part that is Deep-Learning but after a while you will realize that Deep-Learning is actullay large enough to have a entirely seprate look and hence a lot of time when people Say-"Hey mate I am doing Machine Learning!" that should be considered as "Machine Learning without Deep Learning".

So what does this field have-
 - Well this should be looked by several lenses. ( Not only Supervised, Unsupervised etc.)
	 - Defined Problem and Undefined-Problem
	 - Data
	 - Task - ( Supervised, Unsupervised )

 - By Defined Probelm I mean If some work has been done upon previously on this type of task with this  type of data, and mostly the answer is **YES**. If so you works suddenly becomes very less. In startups Data-Scientists work under this category. You have predefined means to approach this probelm.exmples:-

	 - Regression
	 - Classification
	 - Clustering
	 - Recommendation Engine
	 
 - By Undefined Problem I mean research, which could also be divided into 2 types. **This is where hype truly starts**
	- Developing a new algorithm. example - CatBoost was developed, Generative-Pre-Training had same type of data, same task but a new approach.
	- Your task is something new. example - Style-Transfer was entirely new task.
	- Only the CatBosst part was ML others fall under Deep-Learning. 

- Being absolutely honest in **ML data only means tabular data** nothing else(not even a single column with "text"). Anything above that and you need to use Deep-Learning for a lot better results. Also you the dimension of your data shouldn't be too large, if it is  in 1000s then again use Deep-Learning.
- Another aspect of Data is also present but you know/interect with it in later stages and that is a compilation of different data. **Why do you assume you can't have a image and some text corresponding to it along with user information?**. This type of combination is really useful in recommendation systems and some research fields. examples :-
	- Image Summarization
	- Image based question-answering

- **Task? MOSTLY Supervised sometimes Unsupervised** and  I haven't touched Reinforcement Learning till now. Which also makes me conclude that it is safe to assume when some someone says "Hey mate! I am doing ml." he isn't including Reinforcement Learning. And what does Supervised and other things mean you could just goggle it but I hope you know it already.

- Summarizing why it's still used?
	- Its less compute expensive
	- Far more interpretable. **Here comes the math part**

## DL OR Deep-Learning 
This is the hype. A far more complexy, beautiful combination of algorithms and thoughts.
So what does this field have-
 - Well this should also be looked by several lenses. ( Not only Supervised, Unsupervised etc.)
	 - Defined Problem and Undefined-Problem
	 - Data
	 - Task - ( Supervised, Unsupervised )

- The First Point remains the same.
- Data ( Will cover in new section )
- Task? ( Will cover in new section )
- Comparision with ML ( just read )
	- All types of data could be handled
	- Best results
	- Best Transformations of Data can be achieved
	- More compute expensive
	- Less interpretable
	- More complex features can be analysed

## Data
First of all every type of data will get converted to tabular data but the thing is that Machine Learning can't handle high dimensionality of data. For example- "A image from your phone is of about 1920*1080 resolution easily" which makes it very-very large dimensional and not only that if you just make a vector of it you will loose information like crazy.
In case of text too you will have high dimensionality once you convert text to vector.
Other than that data could be classified on basis of content/domain. example-Genomic Data. These types of data contains complex structures hence need complex methods.
These variations in data and the corresponding problem results in specializations in ML corresponing to the data type.

**Hype on basis of Data**

 - Genomic/Bilogical Data could reveal secrets of Human body and solve diseases. Some fruntiers
	 - Genomic Data -> Could Explain everything about us.
	- Neural Imaging Data -> Explains working of Brain.
	- Lung CCTV/MRI Data -> Detection of several types of diseases etc.
	
- Text data is one of the most abundant data and contains lots of varients and problems :-
	- Problems
		- Could be different on the basis of language but same in semantic meaning
		- It need very through cleaning ( You will know once you start dealing with it ).
	- Why helpful?
		- Could explain emotions
		- Question Answer Prediction
		- Generating Literature.

- On Image data there is ( A video is also a set of Images ):-
	- Stype Transfer -> Mix a style of a painting to a photo.
	- Self-Driving Cars
	- When we consider a video some of the new things are:-
		- Removing things/person from a video
		- Video Coloring

- Possibilities:-
	- In the Machine Learning Course by **Andrew Ng** he share a slide about a person gaining some vision after using camers and transfering its signal to the brain through his tounge and that was the moment I choose this field. Later on I found many such beautiful things.
	- Brain-Computer-Interface. (Google it)
	- Personalized Medication/education/everything
	- Neural rendering - You could generate a animated video by describing a scence. ( Not sure whether is exactly lies in neural rendering )
	- And many more.

## Task
- Mostly same as above but I will descibe some froutiers.
- Semi-Supervised and Self-Supervised Learning has come out to a new method approach problems. For example - Given a very large image dataset without labels and one very small dataset with labels we could rotate the images in the large dataset and use that rotation value as target and make a model and then use transfer learning for the smaller dataset which will yield better results.
- GAN's (Generative Adversarial Networks) are now-a-days State-of-Art in terms of training methods and the reason is that they can make synthetic data which not be differentiated from original data by humans and could be used for traning.
	- GAN's are compute expensive as well as they can't be used if you don't have enough data regardless of labels.
- In Unsupervised Learning Deep Learning has some architectures such as RBMs and Deep Belief Networks. These are very useful and fill the gap where GAN's can't be used. For example- In Neuclear Power Plants RBMs (Restricted Boltsman Machines) are used to analysize different and hypothetical possibilites.
- Reinforcement Learning - I don't have any idea so I won't comment.

##  Why math (Interpretability)?

First of all for obvious you need it if you fall under the **For Hype** category. No need to even discuss.
Now for not in that category.
 - Making a model is easy and childs play but 3 questions are not:-
	 -  knowing wheather that model is good? 
	 - it's learning what you want it to? 
	 - how could you improve it?

And to answer these questions you need maths cause **It's less about optimizing your model. It's much more about optimizing your data for your model.** and that comes from knowing your model and that in turn comes from knowing it's math and trust me it's not hard.
One of the major drawback of Neural-Networks is that they are too complex to interpret and hence is also a field of research.

