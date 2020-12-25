# A Deeper Understanding of Linear Regression

When we begin our Machine Learning journey, one of the first algorithms that we come across is linear regression. In this blog I will start things from a beginner level and gradually try to indulge into some deeper aspects of linear regression, which will make you admire the brilliance of this **"simple"** algorithm. So, let's begin!

## Regression problem
A regression problem is where we need to predict a continuous real value for a target variable. A simple example is house price prediction.

## Linear Regression
Given some features ![](https://i.upmath.me/svg/x_%7B1%7D%2C%20x_%7B2%7D......%20x_%7Bd%7D) ( ![](https://i.upmath.me/svg/d) in the dimensionality of the data),  we need to predict a target variable ![](https://i.upmath.me/svg/y). Linear regression proposes to linearly weigh the input features so as to predict the target variable. Mathematically, it can be formulated as

![](https://i.upmath.me/svg/%0A%5Chat%7By%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bi%3Dd%7D%20w_%7Bi%7D%20x_%7Bi%7D%0A)

where ![](https://i.upmath.me/svg/w_%7Bi%7D) is the weight for the ![](https://i.upmath.me/svg/i%5E%7Bth%7D) feature, ![](https://i.upmath.me/svg/%5Chat%7By%7D) is the predicted value

You can observe that the weight vector also has dimensionality ![](https://i.upmath.me/svg/d). Moreover, if both the input features and the weights are represented as vectors ![](https://i.upmath.me/svg/%5Cmathbf%7Bx%7D) and ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D) respectively, they can be expressed as the dot product of the two vectors

![](https://i.upmath.me/svg/%0A%5Chat%7By%7D%20%3D%20%5Cmathbf%7Bw%7D.%5Cmathbf%7Bx%7D%0A)

People often introduce a bias term ![](https://i.upmath.me/svg/b) into the above equations. The bias term is a constant value that is added to get a better prediction for the target variable. It is analogous to the intercept of a straight line in 2-dimensions. For simplicity we will ignore the bias term in our discussion.

The above equations are for a single instance. In machine learning, we encounter dataset with many instances. Hence, there is a need to process multiple instances in one go. We will now convert these operations into a matrix product operation.

Let us assume our dataset has ![](https://i.upmath.me/svg/n) instances, each having ![](https://i.upmath.me/svg/d) features. We represent our data as ![](https://i.upmath.me/svg/%5Cmathbf%7BX%7D) of dimenision (![](https://i.upmath.me/svg/n) x ![](https://i.upmath.me/svg/d)). Our weights of linear regression are represented as ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D), which is a vector of dimension ![](https://i.upmath.me/svg/d). Our output predictions ![](https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7By%7D%7D) can be written as

![](https://i.upmath.me/svg/%0A%5Cmathbf%7B%5Chat%7By%7D%7D%20%3D%20%5Cmathbf%7BX%7D%20%5Cmathbf%7Bw%7D%0A)

We can check that (![](https://i.upmath.me/svg/n) x ![](https://i.upmath.me/svg/d)) matrix multiplied by ![](https://i.upmath.me/svg/d) dimension vector will yield an output of (![](https://i.upmath.me/svg/n) x 1), i.e. 1 value for each of our ![](https://i.upmath.me/svg/n) instances. To introduce the bias term here, a new feature which is always equal to 1 can be added along with the increase in 1 dimension for ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D).

Now, we will look into how we can train our linear regression algorithm. Linear regression is trained using least Mean Square Error (MSE). For data with ![](https://i.upmath.me/svg/n) instances, MSE is defined as :

![](https://i.upmath.me/svg/%0AMSE%20%3D%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bi%3Dn%7D%20(y_%7Bi%7D%20-%20%5Chat%7By%7D_%7Bi%7D)%5E2%0A)

MSE can also be represented in terms of matrix operations:

![](https://i.upmath.me/svg/%0AMSE%20%3D%20%20%5Cfrac%7B1%7D%7Bn%7D%20%7C%7C%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%20%5Cmathbf%7Bw%7D%7C%7C%5E2%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D)%5ET%20(%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D)%0A)

Applying transpose,

![](https://i.upmath.me/svg/%0AMSE%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7By%7D%5ET%20-%20%5Cmathbf%7Bw%7D%5ET%20%5Cmathbf%7BX%7D%5ET)%20(%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D)%0A)

Expanding,

![](https://i.upmath.me/svg/%0AMSE%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%20-%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%20-%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20%2B%20%5Cmathbf%7Bw%7D%5ET%20%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7BX%7D%20%5Cmathbf%7Bw%7D%20)%0A)

Since we need minimal MSE w.r.t. ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D), we need to differentiate MSE w.r.t. ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D) and set the derivative to 0.

![](https://i.upmath.me/svg/%0A%5Cfrac%7B%5Cpartial%20%7BMSE%7D%7D%20%7B%5Cpartial%7B%5Cmathbf%7Bw%7D%7D%7D%20%3D%20-%5Cmathbf%7By%7D%5ET%20%5Cmathbf%7BX%7D%20-%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%20%2B%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%20%2B%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%0A)

Equating derivative to 0, we get

![](https://i.upmath.me/svg/%0A%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%20%3D%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%0A)

Applying transpose to both sides of the equation

![](https://i.upmath.me/svg/%0A%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7By%7D%20%3D%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%20%5Cmathbf%7Bw%7D%0A)
Finally, we obtain the expression of our ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D) vector for which we obtain minimum MSE.

![](https://i.upmath.me/svg/%0A%5Cmathbf%7Bw%7D%20%3D%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)%5E%7B-1%7D%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%0A)

This solution is also called the **closed form** solution for linear regression. The solution for linear regression can also be obtained via an iterative optimization technique, called **Gradient Descent**. We also take a quick look into how it is done.

Our MSE loss equation can be written as,

![](https://i.upmath.me/svg/%0AMSE%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bi%3Dn%7D%20(y_%7Bi%7D%20-%20%5Cmathbf%7Bx_%7Bi%7D%7D%5Cmathbf%7Bw%7D)%5E2%0A)

whose derivative w.r.t. ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D) can be written as

![](https://i.upmath.me/svg/%0A%5Cfrac%7B%5Cpartial%20MSE%7D%7B%5Cpartial%5Cmathbf%7Bw%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bi%3Dn%7D%20-2%20(y_%7Bi%7D%20-%20%5Cmathbf%7Bx_%7Bi%7D%7D%5Cmathbf%7Bw%7D)%20%5Cmathbf%7Bx_%7Bi%7D%7D%0A)

We can verify the above equation: the derivative of a scalar w.r.t a vector is a vector of the same size (in this case ![](https://i.upmath.me/svg/d)). The quantity within parentheses is a scalar quantity multiplied with ![](https://i.upmath.me/svg/%5Cmathbf%7Bx_%7Bi%7D%7D), which is of dimension ![](https://i.upmath.me/svg/d). The weighted vector is then averaged across the data points. This derivative is used to update ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D) as

![](https://i.upmath.me/svg/%0A%5Cmathbf%7Bw%7D%20%20%3A%3D%20%5Cmathbf%7Bw%7D%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%20MSE%7D%7B%5Cpartial%5Cmathbf%7Bw%7D%7D%0A)

where ![](https://i.upmath.me/svg/%5Calpha) is a hyper-parameter called the learning rate. The learning rate adjusts the magnitude of updates that occur to ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D). This updating process is repeated till convergence or for a fixed number of iterations to get the optimal ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D).

## Andrew Ng's drawing of concentric ellipses

If you have seen Andrew Ng's lectures on Linear Regression, you might remember him drawing concentric ellipses and explaining how gradient descent works. Now, we will figure out why he draws ellipses and not any other conic section or shape. **Get ready for some heavy mathematics!!**

Let us represent our MSE loss function as a function of ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D) :

![](https://i.upmath.me/svg/%0AL(%5Cmathbf%7Bw%7D)%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%7C%7C%20%5Cmathbf%7BX%7D%20%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7By%7D%20%7C%7C%20%5E2%0A)

We represent the optimal ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D) as ![](https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D), which is equal to

![](https://i.upmath.me/svg/%0A%5Cmathbf%7B%5Chat%7Bw%7D%7D%20%3D%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)%5E%7B-1%7D%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%0A" )

Let us start exploring the expression

![](https://i.upmath.me/svg/%0A(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%5ET(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A)

![](https://i.upmath.me/svg/(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%5ET) has a shape of (1 x ![](https://i.upmath.me/svg/d)) , ![](https://i.upmath.me/svg/%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D) has shape (![](https://i.upmath.me/svg/d) x ![](https://i.upmath.me/svg/d)) and ![](https://i.upmath.me/svg/(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)) has shape  (![](https://i.upmath.me/svg/d) x ![](https://i.upmath.me/svg/1)). Hence, this expression is a scalar as its shape is (1x1).

Applying transpose,

![](https://i.upmath.me/svg/%0A(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%5ET(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%20%3D%20(%5Cmathbf%7Bw%7D%5ET%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET)(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A)

Expanding the RHS of the equation

![](https://i.upmath.me/svg/%0A%20(%5Cmathbf%7Bw%7D%5ET%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET)(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A)

![](https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%20%2B%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%0A)

Now ![](https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D) and ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D) are same using the property ![](https://i.upmath.me/svg/A%5ET%20%3D%20A) if ![](https://i.upmath.me/svg/A) is scalar. Our RHS becomes

![](https://i.upmath.me/svg/%0A%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%202%20%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%20%2B%20%5Cmathbf%7B%5Chat%7Bw%7D%5ET%7D%20%5Cmathbf%7BX%5ET%7D%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%0A)

From our closed form solution of linear regression,

![](https://i.upmath.me/svg/%0A%5Cmathbf%7B%5Chat%7Bw%7D%7D%20%3D%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)%5E%7B-1%7D%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%0A)

![](https://i.upmath.me/svg/%0A%5Cimplies%20%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%20%3D%20%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7By%7D%0A)

Using the above relation in our RHS, we get

![](https://i.upmath.me/svg/%0A%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%202%20%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%20%2B%20%5Cmathbf%7B%5Chat%7Bw%7D%5ET%7D%20%5Cmathbf%7BX%5ET%7D%5Cmathbf%7By%7D%0A)

Now adding and subtracting ![](https://i.upmath.me/svg/%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D)

![](https://i.upmath.me/svg/%0A%3D%20%5Cunderbrace%7B%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%202%20%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%20%2B%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%20%7D_%7BnL(%5Cmathbf%7Bw%7D)%7D%20-%20%20(%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%5ET%7D%20%5Cmathbf%7BX%5ET%7D%5Cmathbf%7By%7D)%0A)

![](https://i.upmath.me/svg/%0A%3D%20%7C%7C%20%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7By%7D%20%7C%7C%20%5E2%20-%20(%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%20-%20%5Cmathbf%7By%7D%5ET%20%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A)

![](https://i.upmath.me/svg/%0A%3D%20%7C%7C%20%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7By%7D%20%7C%7C%20%5E2%20-%20%5Cmathbf%7By%7D%5ET(%5Cmathbf%7By%7D-%20%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A)
" />

Therefore, we can write our complete equation as

![](https://i.upmath.me/svg/%0A%5Cfrac%7B1%7D%7Bn%7D%20%7C%7C%20%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7By%7D%20%7C%7C%20%5E2%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%5ET%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%20%2B%20%5Cfrac%7B1%7D%7Bn%7D%20%5Cmathbf%7By%7D%5ET(%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A)

Now let us look at what the minimal loss would be as per ![](https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D).

![](https://i.upmath.me/svg/%0AL(%7B%5Cmathbf%7B%5Chat%7Bw%7D%7D)%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%7C%7C%20%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%20-%20%5Cmathbf%7By%7D%20%7C%7C%5E2%20%0A)

![](https://i.upmath.me/svg/%0A%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%20-%20%5Cmathbf%7By%7D)%5ET(%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%20-%20%5Cmathbf%7By%7D)%0A)

![](https://i.upmath.me/svg/%0A%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%20%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7BX%7D%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%20-%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%20%2B%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D)%0A)

We can prove that ![](https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D) = ![](https://i.upmath.me/svg/%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D)

![](https://i.upmath.me/svg/%0A%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%20%3D%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)%5E%7B-1%7D%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%0A)

![](https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)%5E%7B-T%7D%0A)

![](https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D(%5Cmathbf%7BX%7D%5E%7B-1%7D%5Cmathbf%7BX%7D%5E%7B-T%7D)%5ET%0A)

![](https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D(%5Cmathbf%7BX%7D%5E%7B-1%7D%20%5Cmathbf%7BX%7D%5E%7B-T%7D)%0A)

![](https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ET%20%5Cmathbf%7BX%7D%5E%7B-T%7D%0A)

Using the above values,

![](https://i.upmath.me/svg/%0A%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%20%3D%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%5E%7B-T%7D%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%0A)


![](https://i.upmath.me/svg/%0A%20%3D%20%5Cmathbf%7By%7D%5ET%20I%20%5Cmathbf%7By%7D%0A)

![](https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%0A)

Moreover, ![](https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%20%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7BX%7D%20%5Cmathbf%7B%5Chat%7Bw%7D%7D) can be simplified as

![](https://i.upmath.me/svg/%0A%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%20%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7BX%7D%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%20%3D%20%5Cunderbrace%7B%5Cmathbf%7By%7D%5ET%20%5Cmathbf%7BX%7D%5E%7B-T%7D%7D_%7B%5Cmathbf%7B%5Chat%7Bw%7D%5ET%7D%7D%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)%20%5Cunderbrace%7B(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)%5E%7B-1%7D%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%7D_%7B%5Cmathbf%7B%5Chat%7Bw%7D%7D%7D%0A)

![](https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%5E%7B-T%7D%20I%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%0A)

![](https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ETI%20%20%5Cmathbf%7By%7D%0A)

![](https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%0A)

Therefore, we get the following expression for ![](https://i.upmath.me/svg/L(%5Cmathbf%7B%5Chat%7Bw%7D%7D)):

![](https://i.upmath.me/svg/%0AL(%5Cmathbf%7B%5Chat%7Bw%7D%7D)%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%20-%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D)%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Cmathbf%7By%7D%5ET(%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A)

Finally, for the bigger picture we have:

![](https://i.upmath.me/svg/%0A%5Cfrac%7B1%7D%7Bn%7D%20%7C%7C%20%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7By%7D%20%7C%7C%20%5E2%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%5ET%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%20%2B%5Cfrac%7B1%7D%7Bn%7D%20%5Cmathbf%7By%7D%5ET(%5Cmathbf%7By%7D-%20%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A)

![](https://i.upmath.me/svg/%0A%5Cimplies%20L(%5Cmathbf%7Bw%7D)%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%5ET%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%20%2B%20L(%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A)

If we look into the matrix equation of an ellipsoid, we see some similar expression: ![](https://i.upmath.me/svg/v%5ET%5Cmathbf%7BA%7Dv%20%2B%20c%20%3D%200) which represents an ellipsoid if ![](https://i.upmath.me/svg/%5Cmathbf%7BA%7D) is symmetric and positive semi-definite.

A symmetric matrix ![](https://i.upmath.me/svg/%5Cmathbf%7BA%7D) is positive semi-definite if ![](https://i.upmath.me/svg/%5Cforall) non-zero vectors ![](https://i.upmath.me/svg/z), ![](https://i.upmath.me/svg/z%5ET%5Cmathbf%7BA%7Dz%20%3E%3D0).

![](https://i.upmath.me/svg/%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D) is symmetric and we can prove that it is positive semi-definite as follows:

![](https://i.upmath.me/svg/%0Az%5ET%20%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7BX%7D%20z%20%3D%20(%5Cmathbf%7BX%7Dz)%5ET(%5Cmathbf%7BX%7Dz)%20%3D%20%7C%7C%20%5Cmathbf%7BX%7Dz%20%7C%7C%20%5E2%20%3E%3D0%0A)

Hence, ![](https://i.upmath.me/svg/L(%5Cmathbf%7Bw%7D)) can be represented with a ellipse if we are working in 2-dimensions. It must also be noted that the center of the ellipse is ![](https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D). There exist different ellipses for different values of ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D), however each will have its center at ![](https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D). This is the mathematical explanation why Andrew Ng draws concentric ellipses centered at ![](https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D).

## Regularization

The two main regularization for linear regression are Lasso (![](https://i.upmath.me/svg/L1) regularization) and Ridge (![](https://i.upmath.me/svg/L2) regularization). Regularization is a technique which helps us prevent over-fitting by penalizing the complexity of model, i.e. it tries to keep the models simple.

For lasso, the optimization objective becomes ![](https://i.upmath.me/svg/MSE%20%2B%20%5Clambda%7C%7C%5Cmathbf%7Bw%7D%7C%7C) whereas for ridge it comes ![](https://i.upmath.me/svg/MSE%20%2B%20%5Clambda%7C%7C%5Cmathbf%7Bw%7D%7C%7C%5E2), where ![](https://i.upmath.me/svg/%5Clambda) is co-efficient of regularization. We can graphically interpret what the objective tries to do in both the cases. Assuming that we are working in 2-D space, the 2 objectives can be visualized as shown in the figure below.

![Figure 1: Visualizations for Lasso(left) Ridge(right)](assets/LR.jpg)

The constraint of lasso can be converted as ![](https://i.upmath.me/svg/%7C%7C%5Cmathbf%7Bw%7D%7C%7C%20%3C%3D%20k) and of ridge ![](https://i.upmath.me/svg/%7C%7C%5Cmathbf%7Bw%7D%7C%7C%5E2%20%3C%3D%20k). ![](https://i.upmath.me/svg/k) is related to ![](https://i.upmath.me/svg/%5Clambda) here. A larger value of ![](https://i.upmath.me/svg/%5Clambda) will force our ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D) to be smaller, implying a smaller ![](https://i.upmath.me/svg/k). These constraints can be interpreted as a square and a circle for lasso and ridge respectively (as depicted in the above figure). For lasso, we can observe that the the ellipse (our MSE objective) can intersect the square at the axis. If it happens, then one of the components of ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D) **is set exactly to 0**. This implies that there is no contribution of the corresponding input feature in prediction of the target. The intersection at the axis is only possible because of the sharp corners of the square. In the case of ridge, the closest that can happen is one of the components of ![](https://i.upmath.me/svg/%5Cmathbf%7Bw%7D) **being set near to 0, but not exactly 0**. Thus, lasso gives sparse solutions and this property of lasso can be used a **feature selection technique**. The weights of less important features will be reduced to 0 and can be eliminated.

## You have made it to the end!!

In this blog, I have tried to share with you the cool stuff about linear regression that I have come across. I am pretty sure there is much more to linear regression. Do let everyone know about some more interesting things about linear regression in the comment section!
