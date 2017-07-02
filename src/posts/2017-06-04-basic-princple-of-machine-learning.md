
<h1><center>Basic Princple of Machine Learning, <br/>Regressions and Classifications</center></h1>

<center><h4>Ryan J. Kung</h4></center>
<center><h4>ryankung(at)ieee.org<h4></center>


## I Introduction

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

Note that the $hypothesis\ function$ is denoted as $\phi(x)$ in the Deep Learning book.

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

## II, Linear Regression with Multiple Variables

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

## III, Logistic Regression

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

With *sigmoid function* (or *logistic function*, also *activation function* in *neural netowrk* case):

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


## IV Neural Network

* Denotes:

$a_i^{(j)}$= "*activation*" of unit $i$ in layer $j$.

$\Theta^{(j)}$ = matrix of weights controlling function mapping from layer $j$ to layer $j+1$.

If network has $s_j$ units in layer $j$, $s_j+1$ units in layer $j+1$, then $\Theta^{(j)}$ will be of dimension $s_{j+1} \times (s_j+1)$

$L$ = total $no.$ of layer of neural network

$S_l$ = $no.$ of units (not counting bias unit) in layer $l$

$K$ = the number of units in the output layer.

### Cost Function

\begin{gather*} J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*}

Which is similar to Logistic Regression:

\begin{gather*}
J(\theta) = - \frac{1}{m} \sum_{i=1}^m [ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2
\end{gather*}

Note:

* the double sum simply adds up the logistic regression costs calculated for each cell in the output layer
* the triple sum simply adds up the squares of all the individual $\Theta$s in the entire network.
* the $i$ in the triple sum does not refer to training example $i$.

So we Focusing on a single example $x^{(i)}$, $y^{(i)}$, the cause of 1 output unit, and ignore the reqularization ($\lambda=0$)

\begin{gather*}
cost(i) = y^{(i)}\log{h_{\Theta}(x^{(i)}+(1-y^{(i)}logh_{\Theta}(x^{(i)}))}
\end{gather*}

We can think of that $cost(i)\approx (h_{\Theta}(x^{(i)}-y^{i}))^2$ which is $MSE$, for judge how well is the network doing on example $i$.

### Gradient computation

\begin{gather*} J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*}


**Goal**:

\begin{gather*}
\min_\Theta J(\Theta)
\end{gather*}

**Need code to compute**:

\begin{gather*}
J(\Theta)\ {and}\ \dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta)
\end{gather*}


### Backpropagation algorithm:

`Backpropagation` is neural-network terminology for minimizing our cost function, just like what we were doing with gradient descent in logistic and linear regression.


Intuition: $\delta^l_j$ = "error" of cost for $a_j^{(l)}$ (the activation value on node $j$ in layer $l$): 

Formally, $\delta_j^{(l)}=\frac{\partial}{\partial{z_j^{(l)}}}cost(i)$ (for $j \geq 0$), where:

\begin{gather*}
cost(i) = y^{(i)}\log{h_{\Theta}(x^{(i)}+(1-y^{(i)}logh_{\Theta}(x^{(i)}))}
\end{gather*}




##### **In example of applying backpropagation algorithm on a (L=4) Network:**
\begin{gather*}
\delta_j^{(4)}=a_j^{(4)}-y_j=(h_{\Theta}(x))_j-y_i
\end{gather*}


Each of these $\delta_j$, $a_j$ and $y$, is a vector whose dimension is equal to $K$.

\begin{gather*}
\delta^{(3)}=(\Theta^{(3)})^T\delta^{(4)}.*g'(z^{(3)})
\end{gather*}

\begin{gather*}
\delta^{(2)}=(\Theta^{(2)})^T\delta^{(3)}.*g'(z^{(2)})
\end{gather*}


This term $g'(z^{(l)})$, that formally is actually the derivative of the activation function $g$ evaluated at the input values given by $z^{(l)}$. 

**So we can describe the algorithm as:**

Training set ${(x^{(1)}, y^{(1)}),...,(x^{(m)},y^{(m)})}$

Set $\Delta^{(l)}_{ij}=0$ (for all $l, i, j)$

**For** $i=1$ **to** $m$ {

--- Set $a^{(1)} = x^{(i)}$

--- Perform forward proagation to compute $a^{(l)}$ for $l=2,3,...,L$    

--- Using $y^{(i)}$, compute $\delta^{(L)}=a^{(L)}-y^{(i)}$

--- Compute $\delta^{(L-1)},\delta^{(L-2)},...,\delta^{(2)}$    
--- $\Delta^{(l)}_{ij}:=\Delta^{(l)}_ij+a^{(l)}_j\delta^{(l+1)}_i$

}

$D^{(l)}_{ij}:=\frac{1}{m}\Delta^{(l)}_{ij}+\lambda\Theta^{(l)}_{ij}$ if $j \neq 0$

$D^{(l)}_{ij}:=\frac{1}{m}\Delta^{(l)}_{ij}$ if $j = 0$

Finally we got:

\begin{gather*}
\frac{\partial}{\partial{\Theta}_{ij}^{(l)}}J(\Theta)=D^{(l)}_{ij}
\end{gather*}


### Gradient Checking

Gradient checking will assure that our backpropagation works as intended. We can approximate the derivative of our cost function with:

\begin{gather*}
\dfrac{\partial}{\partial\Theta}J(\Theta) \approx \dfrac{J(\Theta + \epsilon) - J(\Theta - \epsilon)}{2\epsilon}
\end{gather*}

With multiple theta matrices, we can approximate the derivative with respect to $\Theta_j$ as follows:

\begin{gather*}
\dfrac{\partial}{\partial\Theta_j}J(\Theta) \approx \dfrac{J(\Theta_1, \dots, \Theta_j + \epsilon, \dots, \Theta_n) - J(\Theta_1, \dots, \Theta_j - \epsilon, \dots, \Theta_n)}{2\epsilon}\end{gather*}

A small value for $\epsilon$ such as $\epsilon=10−4$, guarantees that the math works out properly. If the value for $\epsilon$ is too small, we can end up with numerical problems



Hence, we are only adding or subtracting epsilon to the Θj matrix. In octave we can do it as follows:

```octave
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

We previously saw how to calculate the deltaVector. So once we compute our gradApprox vector, we can check that $gradApprox \approx deltaVector$.

Once you have verified once that your backpropagation algorithm is correct, you don't need to compute gradApprox again. **The code to compute gradApprox can be very slow**.

* [1] What is machine Learning https://www.coursera.org/learn/machine-learning/supplement/aAgxl/what-is-machine-learning
* [2] Supervised Learning https://www.coursera.org/learn/machine-learning/supplement/NKVJ0/supervised-learning
* [3] Unpuservised Learning https://www.coursera.org/learn/machine-learning/supplement/1O0Bk/unsupervised-learning
* [4] Cost Function https://www.coursera.org/learn/machine-learning/supplement/nhzyF/cost-function
* [5] Gradeient descent in parctice I Feature Scaling, https://www.coursera.org/learn/machine-learning/supplement/CTA0D/gradient-descent-in-practice-i-feature-scaling
* [6] Regularized liner regression https://www.coursera.org/learn/machine-learning/lecture/QrMXd/regularized-linear-regression

* [7] Simplified Cost Function https://www.coursera.org/learn/machine-learning/supplement/0hpMl/simplified-cost-function-and-gradient-descent


```octave

```
