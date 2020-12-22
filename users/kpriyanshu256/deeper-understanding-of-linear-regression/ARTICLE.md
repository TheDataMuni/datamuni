# A Deeper Understanding of Linear Regression

When we begin our Machine Learning journey, one of the first algorithms that we come across is linear regression. In this blog I will start things from a beginner level and gradually try to indulge into some deeper aspects of linear regression, which will make you admire the brilliance of this **"simple"** algorithm. So, let's begin!

## Regression problem
A regression problem is where we need to predict a continuous real value for a target variable. A simple example is house price prediction.

## Linear Regression
Given some features <img src="https://i.upmath.me/svg/x_%7B1%7D%2C%20x_%7B2%7D......%20x_%7Bd%7D" alt="x_{1}, x_{2}...... x_{d}" /> ( <img src="https://i.upmath.me/svg/d" alt="d" /> in the dimensionality of the data),  we need to predict a target variable <img src="https://i.upmath.me/svg/y" alt="y" />. Linear regression proposes to linearly weigh the input features so as to predict the target variable. Mathematically, it can be formulated as

<img src="https://i.upmath.me/svg/%0A%5Chat%7By%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bi%3Dd%7D%20w_%7Bi%7D%20x_%7Bi%7D%0A" alt="
\hat{y} = \sum_{i=1}^{i=d} w_{i} x_{i}
" />

where <img src="https://i.upmath.me/svg/w_%7Bi%7D" alt="w_{i}" /> is the weight for the <img src="https://i.upmath.me/svg/i%5E%7Bth%7D" alt="i^{th}" /> feature, <img src="https://i.upmath.me/svg/%5Chat%7By%7D" alt="\hat{y}" /> is the predicted value

You can observe that the weight vector also has dimensionality <img src="https://i.upmath.me/svg/d" alt="d" />. Moreover, if both the input features and the weights are represented as vectors <img src="https://i.upmath.me/svg/%5Cmathbf%7Bx%7D" alt="\mathbf{x}" /> and <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" /> respectively, they can be expressed as the dot product of the two vectors

<img src="https://i.upmath.me/svg/%0A%5Chat%7By%7D%20%3D%20%5Cmathbf%7Bw%7D.%5Cmathbf%7Bx%7D%0A" alt="
\hat{y} = \mathbf{w}.\mathbf{x}
" />

People often introduce a bias term <img src="https://i.upmath.me/svg/b" alt="b" /> into the above equations. The bias term is a constant value that is added to get a better prediction for the target variable. It is analogous to the intercept of a straight line in 2-dimensions. For simplicity we will ignore the bias term in our discussion.

The above equations are for a single instance. In machine learning, we encounter dataset with many instances. Hence, there is a need to process multiple instances in one go. We will now convert these operations into a matrix product operation.

Let us assume our dataset has <img src="https://i.upmath.me/svg/n" alt="n" /> instances, each having <img src="https://i.upmath.me/svg/d" alt="d" /> features. We represent our data as <img src="https://i.upmath.me/svg/%5Cmathbf%7BX%7D" alt="\mathbf{X}" /> of dimenision (<img src="https://i.upmath.me/svg/n" alt="n" /> x <img src="https://i.upmath.me/svg/d" alt="d" />). Our weights of linear regression are represented as <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" />, which is a vector of dimension <img src="https://i.upmath.me/svg/d" alt="d" />. Our output predictions <img src="https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7By%7D%7D" alt="\mathbf{\hat{y}}" /> can be written as

<img src="https://i.upmath.me/svg/%0A%5Cmathbf%7B%5Chat%7By%7D%7D%20%3D%20%5Cmathbf%7BX%7D%20%5Cmathbf%7Bw%7D%0A" alt="
\mathbf{\hat{y}} = \mathbf{X} \mathbf{w}
" />

We can check that (<img src="https://i.upmath.me/svg/n" alt="n" /> x <img src="https://i.upmath.me/svg/d" alt="d" />) matrix multiplied by <img src="https://i.upmath.me/svg/d" alt="d" /> dimension vector will yield an output of (<img src="https://i.upmath.me/svg/n" alt="n" /> x 1), i.e. 1 value for each of our <img src="https://i.upmath.me/svg/n" alt="n" /> instances. To introduce the bias term here, a new feature which is always equal to 1 can be added along with the increase in 1 dimension for <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" />.

Now, we will look into how we can train our linear regression algorithm. Linear regression is trained using least Mean Square Error (MSE). For data with <img src="https://i.upmath.me/svg/n" alt="n" /> instances, MSE is defined as :

<img src="https://i.upmath.me/svg/%0AMSE%20%3D%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bi%3Dn%7D%20(y_%7Bi%7D%20-%20%5Chat%7By%7D_%7Bi%7D)%5E2%0A" alt="
MSE =\frac{1}{n} \sum_{i=1}^{i=n} (y_{i} - \hat{y}_{i})^2
" />

MSE can also be represented in terms of matrix operations:

<img src="https://i.upmath.me/svg/%0AMSE%20%3D%20%20%5Cfrac%7B1%7D%7Bn%7D%20%7C%7C%20%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%20%5Cmathbf%7Bw%7D%7C%7C%5E2%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D)%5ET%20(%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D)%0A" alt="
MSE =  \frac{1}{n} || \mathbf{y} - \mathbf{X} \mathbf{w}||^2 = \frac{1}{n} (\mathbf{y} - \mathbf{X}\mathbf{w})^T (\mathbf{y} - \mathbf{X}\mathbf{w})
" />

Applying transpose,

<img src="https://i.upmath.me/svg/%0AMSE%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7By%7D%5ET%20-%20%5Cmathbf%7Bw%7D%5ET%20%5Cmathbf%7BX%7D%5ET)%20(%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D)%0A" alt="
MSE = \frac{1}{n} (\mathbf{y}^T - \mathbf{w}^T \mathbf{X}^T) (\mathbf{y} - \mathbf{X}\mathbf{w})
" />

Expanding,

<img src="https://i.upmath.me/svg/%0AMSE%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%20-%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%20-%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20%2B%20%5Cmathbf%7Bw%7D%5ET%20%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7BX%7D%20%5Cmathbf%7Bw%7D%20)%0A" alt="
MSE = \frac{1}{n} (\mathbf{y}^T\mathbf{y} - \mathbf{w}^T\mathbf{X}^T\mathbf{y} - \mathbf{y}^T\mathbf{X}\mathbf{w} + \mathbf{w}^T \mathbf{X}^T \mathbf{X} \mathbf{w} )
" />

Since we need minimal MSE w.r.t. <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" />, we need to differentiate MSE w.r.t. <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" /> and set the derivative to 0.

<img src="https://i.upmath.me/svg/%0A%5Cfrac%7B%5Cpartial%20%7BMSE%7D%7D%20%7B%5Cpartial%7B%5Cmathbf%7Bw%7D%7D%7D%20%3D%20-%5Cmathbf%7By%7D%5ET%20%5Cmathbf%7BX%7D%20-%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%20%2B%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%20%2B%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%0A" alt="
\frac{\partial {MSE}} {\partial{\mathbf{w}}} = -\mathbf{y}^T \mathbf{X} - \mathbf{y}^T\mathbf{X} + \mathbf{w}^T\mathbf{X}^T\mathbf{X} + \mathbf{w}^T\mathbf{X}^T\mathbf{X}
" />

Equating derivative to 0, we get

<img src="https://i.upmath.me/svg/%0A%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%20%3D%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%0A" alt="
\mathbf{y}^T\mathbf{X} = \mathbf{w}^T\mathbf{X}^T\mathbf{X}
" />

Applying transpose to both sides of the equation

<img src="https://i.upmath.me/svg/%0A%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7By%7D%20%3D%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%20%5Cmathbf%7Bw%7D%0A" alt="
\mathbf{X}^T \mathbf{y} = \mathbf{X}^T\mathbf{X} \mathbf{w}
" />

Finally, we obtain the expression of our <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" /> vector for which we obtain minimum MSE.

<img src="https://i.upmath.me/svg/%0A%5Cmathbf%7Bw%7D%20%3D%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)%5E%7B-1%7D%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%0A" alt="
\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T\mathbf{y}
" />

This solution is also called the **closed form** solution for linear regression. The solution for linear regression can also be obtained via an iterative optimization technique, called **Gradient Descent**. We also take a quick look into how it is done.

Our MSE loss equation can be written as,

<img src="https://i.upmath.me/svg/%0AMSE%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bi%3Dn%7D%20(y_%7Bi%7D%20-%20%5Cmathbf%7Bx_%7Bi%7D%7D%5Cmathbf%7Bw%7D)%5E2%0A" alt="
MSE = \frac{1}{n} \sum_{i=1}^{i=n} (y_{i} - \mathbf{x_{i}}\mathbf{w})^2
" />

whose derivative w.r.t. <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" /> can be written as

<img src="https://i.upmath.me/svg/%0A%5Cfrac%7B%5Cpartial%20MSE%7D%7B%5Cpartial%5Cmathbf%7Bw%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bi%3Dn%7D%20-2%20(y_%7Bi%7D%20-%20%5Cmathbf%7Bx_%7Bi%7D%7D%5Cmathbf%7Bw%7D)%20%5Cmathbf%7Bx_%7Bi%7D%7D%0A" alt="
\frac{\partial MSE}{\partial\mathbf{w}} = \frac{1}{n} \sum_{i=1}^{i=n} -2 (y_{i} - \mathbf{x_{i}}\mathbf{w}) \mathbf{x_{i}}
" />

We can verify the above equation: the derivative of a scalar w.r.t a vector is a vector of the same size (in this case <img src="https://i.upmath.me/svg/d" alt="d" />). The quantity within parentheses is a scalar quantity multiplied with <img src="https://i.upmath.me/svg/%5Cmathbf%7Bx_%7Bi%7D%7D" alt="\mathbf{x_{i}}" />, which is of dimension <img src="https://i.upmath.me/svg/d" alt="d" />. The weighted vector is then averaged across the data points. This derivative is used to update <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" /> as

<img src="https://i.upmath.me/svg/%0A%5Cmathbf%7Bw%7D%20%20%3A%3D%20%5Cmathbf%7Bw%7D%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%20MSE%7D%7B%5Cpartial%5Cmathbf%7Bw%7D%7D%0A" alt="
\mathbf{w}  := \mathbf{w} - \alpha \frac{\partial MSE}{\partial\mathbf{w}}
" />

where <img src="https://i.upmath.me/svg/%5Calpha" alt="\alpha" /> is a hyper-parameter called the learning rate. The learning rate adjusts the magnitude of updates that occur to <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" />. This updating process is repeated till convergence or for a fixed number of iterations to get the optimal <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" />.

## Andrew Ng's drawing of concentric ellipses

If you have seen Andrew Ng's lectures on Linear Regression, you might remember him drawing concentric ellipses and explaining how gradient descent works. Now, we will figure out why he draws ellipses and not any other conic section or shape. **Get ready for some heavy mathematics!!**

Let us represent our MSE loss function as a function of <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" /> :

<img src="https://i.upmath.me/svg/%0AL(%5Cmathbf%7Bw%7D)%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%7C%7C%20%5Cmathbf%7BX%7D%20%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7By%7D%20%7C%7C%20%5E2%0A" alt="
L(\mathbf{w}) = \frac{1}{n} || \mathbf{X} \mathbf{w} - \mathbf{y} || ^2
" />

We represent the optimal <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" /> as <img src="https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D" alt="\mathbf{\hat{w}}" />, which is equal to

<img src="https://i.upmath.me/svg/%0A%5Cmathbf%7B%5Chat%7Bw%7D%7D%20%3D%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)%5E%7B-1%7D%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%0A" alt="
\mathbf{\hat{w}} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T\mathbf{y}
" />

Let us start exploring the expression

<img src="https://i.upmath.me/svg/%0A(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%5ET(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A" alt="
(\mathbf{w} - \mathbf{\hat{w}})^T(\mathbf{X}^T\mathbf{X})(\mathbf{w} - \mathbf{\hat{w}})
" />

<img src="https://i.upmath.me/svg/(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%5ET" alt="(\mathbf{w} - \mathbf{\hat{w}})^T" /> has a shape of (1 x <img src="https://i.upmath.me/svg/d" alt="d" />) , <img src="https://i.upmath.me/svg/%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D" alt="\mathbf{X}^T\mathbf{X}" /> has shape (<img src="https://i.upmath.me/svg/d" alt="d" /> x <img src="https://i.upmath.me/svg/d" alt="d" />) and <img src="https://i.upmath.me/svg/(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)" alt="(\mathbf{w} - \mathbf{\hat{w}})" /> has shape  (<img src="https://i.upmath.me/svg/d" alt="d" /> x <img src="https://i.upmath.me/svg/1" alt="1" />). Hence, this expression is a scalar as its shape is (1x1).

Applying transpose,

<img src="https://i.upmath.me/svg/%0A(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%5ET(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%20%3D%20(%5Cmathbf%7Bw%7D%5ET%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET)(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A" alt="
(\mathbf{w} - \mathbf{\hat{w}})^T(\mathbf{X}^T\mathbf{X})(\mathbf{w} - \mathbf{\hat{w}}) = (\mathbf{w}^T - \mathbf{\hat{w}}^T)(\mathbf{X}^T\mathbf{X})(\mathbf{w} - \mathbf{\hat{w}})
" />

Expanding the RHS of the equation

<img src="https://i.upmath.me/svg/%0A%20(%5Cmathbf%7Bw%7D%5ET%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET)(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A" alt="
 (\mathbf{w}^T - \mathbf{\hat{w}}^T)(\mathbf{X}^T\mathbf{X}\mathbf{w} - \mathbf{X}^T\mathbf{X}\mathbf{\hat{w}})
" />

<img src="https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%20%2B%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%0A" alt="
= \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w} - \mathbf{\hat{w}}^T\mathbf{X}^T\mathbf{X}\mathbf{w} - \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{\hat{w}} + \mathbf{\hat{w}}^T\mathbf{X}^T\mathbf{X}\mathbf{\hat{w}}
" />

Now <img src="https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D" alt="\mathbf{\hat{w}}^T\mathbf{X}^T\mathbf{X}\mathbf{w}" /> and <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D" alt="\mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{\hat{w}}" /> are same using the property <img src="https://i.upmath.me/svg/A%5ET%20%3D%20A" alt="A^T = A" /> if <img src="https://i.upmath.me/svg/A" alt="A" /> is scalar. Our RHS becomes

<img src="https://i.upmath.me/svg/%0A%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%202%20%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%20%2B%20%5Cmathbf%7B%5Chat%7Bw%7D%5ET%7D%20%5Cmathbf%7BX%5ET%7D%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%0A" alt="
 \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w} - 2  \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{\hat{w}} + \mathbf{\hat{w}^T} \mathbf{X^T}\mathbf{X}\mathbf{\hat{w}}
" />

From our closed form solution of linear regression,

<img src="https://i.upmath.me/svg/%0A%5Cmathbf%7B%5Chat%7Bw%7D%7D%20%3D%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)%5E%7B-1%7D%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%0A" alt="
\mathbf{\hat{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
" />

<img src="https://i.upmath.me/svg/%0A%5Cimplies%20%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%20%3D%20%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7By%7D%0A" alt="
\implies  \mathbf{X}^T\mathbf{X} \mathbf{\hat{w}} = \mathbf{X}^T \mathbf{y}
" />

Using the above relation in our RHS, we get

<img src="https://i.upmath.me/svg/%0A%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%202%20%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%20%2B%20%5Cmathbf%7B%5Chat%7Bw%7D%5ET%7D%20%5Cmathbf%7BX%5ET%7D%5Cmathbf%7By%7D%0A" alt="
\mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w} - 2  \mathbf{w}^T\mathbf{X}^T\mathbf{y} + \mathbf{\hat{w}^T} \mathbf{X^T}\mathbf{y}
" />

Now adding and subtracting <img src="https://i.upmath.me/svg/%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D" alt="\mathbf{y}^T\mathbf{y}" />

<img src="https://i.upmath.me/svg/%0A%3D%20%5Cunderbrace%7B%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%202%20%20%5Cmathbf%7Bw%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%20%2B%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%20%7D_%7BnL(%5Cmathbf%7Bw%7D)%7D%20-%20%20(%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%5ET%7D%20%5Cmathbf%7BX%5ET%7D%5Cmathbf%7By%7D)%0A" alt="
= \underbrace{\mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w} - 2  \mathbf{w}^T\mathbf{X}^T\mathbf{y} + \mathbf{y}^T\mathbf{y} }_{nL(\mathbf{w})} -  (\mathbf{y}^T\mathbf{y} - \mathbf{\hat{w}^T} \mathbf{X^T}\mathbf{y})
" />

<img src="https://i.upmath.me/svg/%0A%3D%20%7C%7C%20%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7By%7D%20%7C%7C%20%5E2%20-%20(%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%20-%20%5Cmathbf%7By%7D%5ET%20%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A" alt="
= || \mathbf{X}\mathbf{w} - \mathbf{y} || ^2 - (\mathbf{y}^T\mathbf{y} - \mathbf{y}^T \mathbf{X}\mathbf{\hat{w}})
" />

<img src="https://i.upmath.me/svg/%0A%3D%20%7C%7C%20%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7By%7D%20%7C%7C%20%5E2%20-%20%5Cmathbf%7By%7D%5ET(%5Cmathbf%7By%7D-%20%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A" alt="
= || \mathbf{X}\mathbf{w} - \mathbf{y} || ^2 - \mathbf{y}^T(\mathbf{y}- \mathbf{X}\mathbf{\hat{w}})
" />

Therefore, we can write our complete equation as

<img src="https://i.upmath.me/svg/%0A%5Cfrac%7B1%7D%7Bn%7D%20%7C%7C%20%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7By%7D%20%7C%7C%20%5E2%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%5ET%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%20%2B%20%5Cfrac%7B1%7D%7Bn%7D%20%5Cmathbf%7By%7D%5ET(%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A" alt="
\frac{1}{n} || \mathbf{X}\mathbf{w} - \mathbf{y} || ^2 = \frac{1}{n} (\mathbf{w} - \mathbf{\hat{w}})^T (\mathbf{X}^T\mathbf{X})(\mathbf{w} - \mathbf{\hat{w}}) + \frac{1}{n} \mathbf{y}^T(\mathbf{y} - \mathbf{X}\mathbf{\hat{w}})
" />

Now let us look at what the minimal loss would be as per <img src="https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D" alt="\mathbf{\hat{w}}" />.

<img src="https://i.upmath.me/svg/%0AL(%7B%5Cmathbf%7B%5Chat%7Bw%7D%7D)%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%7C%7C%20%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%20-%20%5Cmathbf%7By%7D%20%7C%7C%5E2%20%0A" alt="
L({\mathbf{\hat{w}}) = \frac{1}{n} || \mathbf{X}\mathbf{\hat{w}} - \mathbf{y} ||^2
" />

<img src="https://i.upmath.me/svg/%0A%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%20-%20%5Cmathbf%7By%7D)%5ET(%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%20-%20%5Cmathbf%7By%7D)%0A" alt="
 = \frac{1}{n} (\mathbf{X}\mathbf{\hat{w}} - \mathbf{y})^T(\mathbf{X}\mathbf{\hat{w}} - \mathbf{y})
" />

<img src="https://i.upmath.me/svg/%0A%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%20%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7BX%7D%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%20-%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%20%2B%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D)%0A" alt="
 = \frac{1}{n} (\mathbf{\hat{w}}^T \mathbf{X}^T \mathbf{X} \mathbf{\hat{w}} - \mathbf{y}^T\mathbf{X}\mathbf{\hat{w}} - \mathbf{\hat{w}}^T\mathbf{X}^T\mathbf{y} + \mathbf{y}^T\mathbf{y})
" />

We can prove that <img src="https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D" alt="\mathbf{\hat{w}}^T\mathbf{X}^T\mathbf{y}" /> = <img src="https://i.upmath.me/svg/%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D" alt="\mathbf{y}^T\mathbf{y}" />

<img src="https://i.upmath.me/svg/%0A%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%20%3D%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)%5E%7B-1%7D%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%0A" alt="
\mathbf{\hat{w}}^T = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
" />

<img src="https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)%5E%7B-T%7D%0A" alt="
= \mathbf{y}^T\mathbf{X} (\mathbf{X}^T\mathbf{X})^{-T}
" />

<img src="https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D(%5Cmathbf%7BX%7D%5E%7B-1%7D%5Cmathbf%7BX%7D%5E%7B-T%7D)%5ET%0A" alt="
= \mathbf{y}^T\mathbf{X}(\mathbf{X}^{-1}\mathbf{X}^{-T})^T
" />

<img src="https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D(%5Cmathbf%7BX%7D%5E%7B-1%7D%20%5Cmathbf%7BX%7D%5E%7B-T%7D)%0A" alt="
= \mathbf{y}^T\mathbf{X}(\mathbf{X}^{-1} \mathbf{X}^{-T})
" />

<img src="https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ET%20%5Cmathbf%7BX%7D%5E%7B-T%7D%0A" alt="
= \mathbf{y}^T \mathbf{X}^{-T}
" />

Using the above values,

<img src="https://i.upmath.me/svg/%0A%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%20%3D%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%5E%7B-T%7D%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%0A" alt="
\mathbf{\hat{w}}^T\mathbf{X}^T\mathbf{y} = \mathbf{y}^T\mathbf{X}^{-T} \mathbf{X}^T\mathbf{y}
" />


<img src="https://i.upmath.me/svg/%0A%20%3D%20%5Cmathbf%7By%7D%5ET%20I%20%5Cmathbf%7By%7D%0A" alt="
 = \mathbf{y}^T I \mathbf{y}
" />

<img src="https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%0A" alt="
= \mathbf{y}^T\mathbf{y}
" />

Moreover, <img src="https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%20%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7BX%7D%20%5Cmathbf%7B%5Chat%7Bw%7D%7D" alt="\mathbf{\hat{w}}^T \mathbf{X}^T \mathbf{X} \mathbf{\hat{w}}" /> can be simplified as

<img src="https://i.upmath.me/svg/%0A%5Cmathbf%7B%5Chat%7Bw%7D%7D%5ET%20%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7BX%7D%20%5Cmathbf%7B%5Chat%7Bw%7D%7D%20%3D%20%5Cunderbrace%7B%5Cmathbf%7By%7D%5ET%20%5Cmathbf%7BX%7D%5E%7B-T%7D%7D_%7B%5Cmathbf%7B%5Chat%7Bw%7D%5ET%7D%7D%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)%20%5Cunderbrace%7B(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)%5E%7B-1%7D%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%7D_%7B%5Cmathbf%7B%5Chat%7Bw%7D%7D%7D%0A" alt="
\mathbf{\hat{w}}^T \mathbf{X}^T \mathbf{X} \mathbf{\hat{w}} = \underbrace{\mathbf{y}^T \mathbf{X}^{-T}}_{\mathbf{\hat{w}^T}} (\mathbf{X}^T\mathbf{X}) \underbrace{(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}_{\mathbf{\hat{w}}}
" />

<img src="https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%5E%7B-T%7D%20I%20%5Cmathbf%7BX%7D%5ET%5Cmathbf%7By%7D%0A" alt="
= \mathbf{y}^T\mathbf{X}^{-T} I \mathbf{X}^T\mathbf{y}
" />

<img src="https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ETI%20%20%5Cmathbf%7By%7D%0A" alt="
= \mathbf{y}^TI  \mathbf{y}
" />

<img src="https://i.upmath.me/svg/%0A%3D%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%0A" alt="
= \mathbf{y}^T\mathbf{y}
" />

Therefore, we get the following expression for <img src="https://i.upmath.me/svg/L(%5Cmathbf%7B%5Chat%7Bw%7D%7D)" alt="L(\mathbf{\hat{w}})" />:

<img src="https://i.upmath.me/svg/%0AL(%5Cmathbf%7B%5Chat%7Bw%7D%7D)%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7By%7D%5ET%5Cmathbf%7By%7D%20-%20%5Cmathbf%7By%7D%5ET%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D)%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Cmathbf%7By%7D%5ET(%5Cmathbf%7By%7D%20-%20%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A" alt="
L(\mathbf{\hat{w}}) = \frac{1}{n} (\mathbf{y}^T\mathbf{y} - \mathbf{y}^T\mathbf{X}\mathbf{\hat{w}}) = \frac{1}{n} \mathbf{y}^T(\mathbf{y} - \mathbf{X}\mathbf{\hat{w}})
" />

Finally, for the bigger picture we have:

<img src="https://i.upmath.me/svg/%0A%5Cfrac%7B1%7D%7Bn%7D%20%7C%7C%20%5Cmathbf%7BX%7D%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7By%7D%20%7C%7C%20%5E2%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%5ET%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%20%2B%5Cfrac%7B1%7D%7Bn%7D%20%5Cmathbf%7By%7D%5ET(%5Cmathbf%7By%7D-%20%5Cmathbf%7BX%7D%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A" alt="
\frac{1}{n} || \mathbf{X}\mathbf{w} - \mathbf{y} || ^2 = \frac{1}{n} (\mathbf{w} - \mathbf{\hat{w}})^T (\mathbf{X}^T\mathbf{X})(\mathbf{w} - \mathbf{\hat{w}}) +\frac{1}{n} \mathbf{y}^T(\mathbf{y}- \mathbf{X}\mathbf{\hat{w}})
" />

<img src="https://i.upmath.me/svg/%0A%5Cimplies%20L(%5Cmathbf%7Bw%7D)%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%5ET%20(%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D)(%5Cmathbf%7Bw%7D%20-%20%5Cmathbf%7B%5Chat%7Bw%7D%7D)%20%2B%20L(%5Cmathbf%7B%5Chat%7Bw%7D%7D)%0A" alt="
\implies L(\mathbf{w}) = \frac{1}{n} (\mathbf{w} - \mathbf{\hat{w}})^T (\mathbf{X}^T\mathbf{X})(\mathbf{w} - \mathbf{\hat{w}}) + L(\mathbf{\hat{w}})
" />

If we look into the matrix equation of an ellipsoid, we see some similar expression: <img src="https://i.upmath.me/svg/v%5ET%5Cmathbf%7BA%7Dv%20%2B%20c%20%3D%200" alt="v^T\mathbf{A}v + c = 0" /> which represents an ellipsoid if <img src="https://i.upmath.me/svg/%5Cmathbf%7BA%7D" alt="\mathbf{A}" /> is symmetric and positive semi-definite.

A symmetric matrix <img src="https://i.upmath.me/svg/%5Cmathbf%7BA%7D" alt="\mathbf{A}" /> is positive semi-definite if <img src="https://i.upmath.me/svg/%5Cforall" alt="\forall" /> non-zero vectors <img src="https://i.upmath.me/svg/z" alt="z" />, <img src="https://i.upmath.me/svg/z%5ET%5Cmathbf%7BA%7Dz%20%3E%3D0" alt="z^T\mathbf{A}z &gt;=0" />.

<img src="https://i.upmath.me/svg/%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D" alt="\mathbf{X}^T\mathbf{X}" /> is symmetric and we can prove that it is positive semi-definite as follows:

<img src="https://i.upmath.me/svg/%0Az%5ET%20%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7BX%7D%20z%20%3D%20(%5Cmathbf%7BX%7Dz)%5ET(%5Cmathbf%7BX%7Dz)%20%3D%20%7C%7C%20%5Cmathbf%7BX%7Dz%20%7C%7C%20%5E2%20%3E%3D0%0A" alt="
z^T \mathbf{X}^T \mathbf{X} z = (\mathbf{X}z)^T(\mathbf{X}z) = || \mathbf{X}z || ^2 &gt;=0
" />

Hence, <img src="https://i.upmath.me/svg/L(%5Cmathbf%7Bw%7D)" alt="L(\mathbf{w})" /> can be represented with a ellipse if we are working in 2-dimensions. It must also be noted that the center of the ellipse is <img src="https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D" alt="\mathbf{\hat{w}}" />. There exist different ellipses for different values of <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" />, however each will have its center at <img src="https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D" alt="\mathbf{\hat{w}}" />. This is the mathematical explanation why Andrew Ng draws concentric ellipses centered at <img src="https://i.upmath.me/svg/%5Cmathbf%7B%5Chat%7Bw%7D%7D" alt="\mathbf{\hat{w}}" />.

## Regularization

The two main regularization for linear regression are Lasso (<img src="https://i.upmath.me/svg/L1" alt="L1" /> regularization) and Ridge (<img src="https://i.upmath.me/svg/L2" alt="L2" /> regularization). Regularization is a technique which helps us prevent over-fitting by penalizing the complexity of model, i.e. it tries to keep the models simple.

For lasso, the optimization objective becomes <img src="https://i.upmath.me/svg/MSE%20%2B%20%5Clambda%7C%7C%5Cmathbf%7Bw%7D%7C%7C" alt="MSE + \lambda||\mathbf{w}||" /> whereas for ridge it comes <img src="https://i.upmath.me/svg/MSE%20%2B%20%5Clambda%7C%7C%5Cmathbf%7Bw%7D%7C%7C%5E2" alt="MSE + \lambda||\mathbf{w}||^2" />, where <img src="https://i.upmath.me/svg/%5Clambda" alt="\lambda" /> is co-efficient of regularization. We can graphically interpret what the objective tries to do in both the cases. Assuming that we are working in 2-D space, the 2 objectives can be visualized as shown in the figure below.

![Figure 1: Visualizations for Lasso(left) Ridge(right)](assets/LR.jpg)

The constraint of lasso can be converted as <img src="https://i.upmath.me/svg/%7C%7C%5Cmathbf%7Bw%7D%7C%7C%20%3C%3D%20k" alt="||\mathbf{w}|| &lt;= k" /> and of ridge <img src="https://i.upmath.me/svg/%7C%7C%5Cmathbf%7Bw%7D%7C%7C%5E2%20%3C%3D%20k" alt="||\mathbf{w}||^2 &lt;= k" />. <img src="https://i.upmath.me/svg/k" alt="k" /> is related to <img src="https://i.upmath.me/svg/%5Clambda" alt="\lambda" /> here. A larger value of <img src="https://i.upmath.me/svg/%5Clambda" alt="\lambda" /> will force our <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" /> to be smaller, implying a smaller <img src="https://i.upmath.me/svg/k" alt="k" />. These constraints can be interpreted as a square and a circle for lasso and ridge respectively (as depicted in the above figure). For lasso, we can observe that the the ellipse (our MSE objective) can intersect the square at the axis. If it happens, then one of the components of <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" /> **is set exactly to 0**. This implies that there is no contribution of the corresponding input feature in prediction of the target. The intersection at the axis is only possible because of the sharp corners of the square. In the case of ridge, the closest that can happen is one of the components of <img src="https://i.upmath.me/svg/%5Cmathbf%7Bw%7D" alt="\mathbf{w}" /> **being set near to 0, but not exactly 0**. Thus, lasso gives sparse solutions and this property of lasso can be used a **feature selection technique**. The weights of less important features will be reduced to 0 and can be eliminated.

## You have made it to the end!!

In this blog, I have tried to share with you the cool stuff about linear regression that I have come across. I am pretty sure there is much more to linear regression. Do let everyone know about some more interesting things about linear regression in the comment section!
