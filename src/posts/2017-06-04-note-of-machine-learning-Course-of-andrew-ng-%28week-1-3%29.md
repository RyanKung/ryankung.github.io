
# Note of Machine Learning Course of Andrew Ng (week 1-3)

<center><h4>Ryan J. Kung</h4></center>
<center><h4>ryankung(at)ieee.org<h4></center>

## Week 1, Introduction

### 1.1 What is Machine Learning[1]

Two definitions of Machine Learning are offered:

*Arthur Samuel* described it as: "**the field of study that gives computers the ability to learn without being explicitly programmed.**" This is an older, informal definition.

*Tom Mitchell* provides a more modern definition: "**A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.**"

### 1.1.1 Supervised Learning and Unsupervised Learning

#### Supervised Learning[2]

* Regression

* Classification

#### Unsupervised Learning[3]

* Clustering

* Non-clustering

### 1.2 Model and Cost Function

##### Input and Output

* $x^{(i)}$ denotes $input$ variable or feature, $y^{(i)}$ denotes $output$.

* $(x^{(i)}, y^{(i)})\ |\  i \in [1, m],\ i \in \mathbb{R}$ is called $Training\ Sample$.

* $(X, Y)$ is the $space$ of $(Input, Output)$ valuses, which $X=Y=\mathbb{R}$

#### Hypothesis Function

* **Goal**: Given a Training Set, to Learn a Function $h: X \rightarrow Y$, So that $h(x)$ is a $good$ predictor for the corresponding value $y$.

#### Cost Function[4]

* Measure the $Hypothesis\ Function$. Denoted as $J(\theta)$.

* **Idea**: Choose $\theta$ so that $h_{\theta}$ is close to $y$ for training samples $(X, Y)$.


##### Cost Function :: MSE

This function is otherwise called the "Squared error function", or "Mean squared error". 

\begin{equation}
J(\theta_0, \theta_1) = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left ( \hat{y}_{i}- y_{i} \right)^2 = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left (h_\theta (x_{i}) - y_{i} \right)^2
\end{equation}

### 1.3 Gradient Descent

For minimizing the cost Function Z, We keep changing $\theta$ to reduce $J(\theta)$.

repeat until convergance {
\begin{equation}
\theta_i := \theta_i - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
\end{equation}

}

Where $\alpha$ is the learning rate. And $\theta_{1..k}$ should be update simultaneously.

* Convex Function means Bowl-shaped Function which dosen't have any local optima except for the one global optimum.

* "Batch" Gradient Decent: Each step of gradient decent uses all the training examples.

## Week 2, Linear Regression with Multiple Variables

### 2.1 Multi Features


Denotes:

* $n$: $\left| x^{(i)} \right|$ Number of features

* $x^{(i)}$: Input (features) vactor of $i^{th}$ training example

* $x_j^{(i)}$: Value of feature $j$ in $i^{th}$ traning example.

Hypothesis:

We suppose that:

\begin{align*}
x_0^{(i)}=1\\
\end{align*}

So:

\begin{align*}
h_{\theta} &= \theta_0 + \theta_1x_1 + \theta_2x_2 +...+\theta_kx_k\\
&=\theta_0x_0 + \theta_1x_1 + \theta_2x_2 +...+\theta_kx_k \\&=\begin{bmatrix}\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}\\ &= \Theta^T X
\end{align*}

#### Feature Scaling

* Idea: Make sure features are on a similar scale. Thus to get every feature into **approximately** $x_i \in [-1, 1]$

* Mean Normalization:

\begin{align*}
x_i := \dfrac{x_i - \mu_i}{s_i}
\end{align*}

   Where $\mu_i$ is the **average** of all values for feature(i), and $s_i$ is the range value of values $(max-min)$, or $s_i$ is the standard deviation.[5]


#### Learning Rate



Measure $J(\theta, iteration)$ to makeing sure gradient descent working correctly. $J(\theta, iteratorn)$ should be a $convex\ function$.

- If $\alpha$ is too small: Slow convergence.
- if $\alpha$ is too large: $J(\theta)$ may not decrese on every iteration; may not converge.

To choose $\alpha$ try:

<center>..., 0.001, 0.01, .., 0.1, 1, ...</center>

#### Polynomial Regression

**Polynomial Regression**

Our hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can change **the behavior or curve** of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

*Linear Function -> Quadratic Function -> Cubic Function -> ...*



### 2.2 Computing Parameters Analytically

#### Normal Equation

*Normal Equation*: Method to solve for $\theta$ analytically.

* Intution: If 1D ($\theta \in \mathbb{R}$)
\begin{align*}
J(\theta)=\alpha \theta^2 + b\theta + c
\end{align*}

To Set: 
\begin{align*}
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{d}{d \theta_j} J(\theta) = 0
\end{align*}


Solve for $\theta$

* If nD ($\theta \in \mathbb{R}^{n+1}$), A Vertex.

\begin{align*}
J(\Theta)=\frac{1}{2m} \sum_{i=1}^m (h_{\theta}(x^{(i)}-y^{(i)})^2
\end{align*}

To Set: 
\begin{align*}
\frac{\partial}{\partial \theta_j} J(\theta) = 0
\end{align*}

(for every $j$)
 
Solve for $\Theta$

* Solution:
\begin{align*}
\Theta =(X^TX)^{-1}X^TY
\end{align*}

* Octave Code: 

<center>`pinv(X'*X)*X'*Y`</center>


The following is a comparison of gradient descent and the normal equation:

|Gradient Descent	|Normal Equation|
| :------------- |-------------:|
|Need to choose $\alpha$	|No need to choose $\alpha$|
|Needs many iterations	|No need to iterate|
|$O (kn2)$	            |$O (n3)$, need to calculate inverse of $X^TX^{-1}$|
|Works well when $n$ is large	|Slow if $n$ is very large|


#### Normal Equation Non-inveribility

If $X^TX$ is non-invertible.

* Redundant features(linearly dependent).
* Too many features (e.g. $m\leq m$): Delete some feature, or use **regularization**.

## Week3, Logistic Regression

### 3.1 Classification and Representation

#### Classification

$y \in \{0, 1\}$

*Threshold classifier output $h_{\theta}(x)$ at 0.5.*


if $h_{\theta}(x) \geq 0.5$, predict "$y=1$"

if $h_{\theta}(x) \leq 0.5$, predict "$y=0$"

Classification: $y=0, 1$

$h(x)$ can be >1 or <0

#### Hypothesis Representation

Logistic Regression Model

Want 

\begin{align*}
h_{\theta}(x) \in [0, 1]
\end{align*}


\begin{align*}
h_{\theta}(x)=g(\theta^Tx)
\end{align*}

With *sigmoid function* (or *logistic function*):

\begin{align*}
g(z)=\frac{1}{1+e^{-z}}\\
h_{\theta}(x)=\frac{1}{1+e^{-\theta^Tx}}
\end{align*}



* Interpretation of Hypothesis Output

\begin{align*}
h_{\theta}(x)=P(y=1\ |\ x;\theta)
\end{align*}


#### Decision Boundary




* Suppose 

predict "$y=1$" if $h_{\theta}\geq0.5$

predict "$y=0$" if $h_{\theta}\lt0.5$

* Then

$g(z)\geq0.5$ when $z\geq0$

$h_{\theta}(x)=g(\theta^Tx)\geq0.5$ wherever $\theta^Tx\geq 0$


* So

Predict $y=1$ if $\theta^Tx\geq 0$  


* The line of $\theta^Tx = 0$ is called **decision boundary**.

#### Cost Function

\begin{align*}
J(\theta) = \sum_{i=1}^m Cost(h_{\theta}(x^{(i)}), y)
\end{align*}

If we use the linear regression cost function as cost functon of logistic regression, It would be none-convex function of $\theta$

Logisitc regression cost function:

\begin{align*}& J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1} \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; & \text{if y = 0}\end{align*}

Which can be rewrite as:

\begin{align*}
J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]
\end{align*}

This cost function can be derived from statistics using the principle of **maximum likelihood estimation**[7]. 

#### Advanced Optimization

"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize θ that can be used instead of gradient descent. We suggest that you should not write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized. Octave provides them.

### 3.2 Multiclass Classification

* One-Vs-All (one-vs-rest) Classification

To train datasets with binary logisic classification as the binary tree

\begin{align*}& y \in \lbrace0, 1 ... n\rbrace \newline& h_\theta^{(0)}(x) = P(y = 0 | x ; \theta) \newline& h_\theta^{(1)}(x) = P(y = 1 | x ; \theta) \newline& \cdots \newline& h_\theta^{(n)}(x) = P(y = n | x ; \theta) \newline& \mathrm{prediction} = \max_i( h_\theta ^{(i)}(x) )\newline\end{align*}


### 3.3 The Problem of Overfitting


* underfitting -> have high bias

* overfitting -> has high variance

`overfitting` is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

 There are two main options to address the issue of overfitting:

* **Reduce the number of features:**

    Manually select which features to keep.
    Use a model selection algorithm (studied later in the course).
    
    

* **Regularization**

    Keep all the features, but reduce the magnitude of parameters θj.
    Regularization works well when we have a lot of slightly useful features.
    
    Small values for parameters $\theta_0, \theta_1...,\theta_n$
    
    - 'simpler' hypothesis
    - Less prone to overfitting
    
    

Since that we have a lot of features(a hundred maybe), and dont know select which $\theta$ to shrink, so the basic idea is to do summation for all $\theta_{1,n}$.

Note that we should not shrink $\theta_0$, which make very little difference to the result.

\begin{align*}
min_\theta\ \dfrac{1}{2m}\  \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2
\end{align*}


The λ, or lambda, is the **regularization parameter**. It determines how much the costs of our theta parameters are inflated.

* Goals of $Lambda$:

    - The first goal, capture it by the first goal objective, is that we would like to train, is that we would like to fit the training data well. We would like to fit the training set well. 

    - The second goal is, we want to keep the parameters small, and that's captured by the second term, by the regularization objective. 

#### Regularized Linear Regression

##### On Gradient decent

\begin{align*} & \text{Repeat}\ \lbrace \newline & \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline & \rbrace\\ \end{align*}

And $\theta_j$ can represent with: $\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$

The first term in the above equation, $1 - \alpha\frac{\lambda}{m}$ will usually less than 1, because alpha times lambda over m is going to be positive, and usually if your learning rate is small and if m is large, this is usually pretty small[6]. 

Address to $1 - \alpha\frac{\lambda}{m}$ is less than 1, through the iteration, the $\theta_j$ will become smaller and smaller.

##### On Normal Equation

\begin{align*}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \newline& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \newline & 1 & & & \newline & & 1 & & \newline & & & \ddots & \newline & & & & 1 \newline\end{bmatrix}\end{align*}

Note that $L$ is an n-1 x n-1 matrix.

If m < n, then $X^TX$ is non-invertible. However, when we add the term λ⋅L, then $X^TX + λ⋅L$ becomes invertible.


#### Regularized Logistic Regression

##### Cost Function
\begin{align*}
J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2
\end{align*}

#### Gradient Descent

\begin{align*} & \text{Repeat}\ \lbrace \newline & \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline & \rbrace\\ \end{align*}


* [1] What is machine Learning https://www.coursera.org/learn/machine-learning/supplement/aAgxl/what-is-machine-learning
* [2] Supervised Learning https://www.coursera.org/learn/machine-learning/supplement/NKVJ0/supervised-learning
* [3] Unpuservised Learning https://www.coursera.org/learn/machine-learning/supplement/1O0Bk/unsupervised-learning
* [4] Cost Function https://www.coursera.org/learn/machine-learning/supplement/nhzyF/cost-function
* [5] Gradeient descent in parctice I Feature Scaling, https://www.coursera.org/learn/machine-learning/supplement/CTA0D/gradient-descent-in-practice-i-feature-scaling
* [6] Regularized liner regression https://www.coursera.org/learn/machine-learning/lecture/QrMXd/regularized-linear-regression

* [7] Simplified Cost Function https://www.coursera.org/learn/machine-learning/supplement/0hpMl/simplified-cost-function-and-gradient-descent


```octave

```
