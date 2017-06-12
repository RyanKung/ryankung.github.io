
<h1><center>Multi-Layer Neural Networks</center></h1>

<center>**Ryan J. Kung**</center>
<center>**ryankung(at)ieee.org**</center>


### Deep Feedforward Networks

$Deep\  feedforward\ networks$, also often called $feedforward \ neural \ network$, or $multilayer\ percptrons$($MLPs$), are the quientessential deep learning models.

**Goal:** Approximate some function $f*$. in classifier case, $f^*(x)$ maps an input $x$ to a category $y$, It defaines a mapping $y=f(x;\theta)$ and learns the value of the parameters(weight) $\theta$ that result in best function approximation.

* Forward Propagation:

information flows through the function being evalutied from $x$, through the intermediate computations used to define $f$, and finally to the output $y$. **There are no $feedback$ connections in which outputs of model are fed back to it self**.

* Network

Represented by composing together many different functions. The model is associated with a directed **acyclic graph** describing how functions are compose together.

We meight have functions $f^{(1)}(x),f^{(2)}(x),..,f^{(n)}(x)$, connected in a chain, wich can form as $f^{(1)}\circ f^{(2)}\circ..f^{(n)}(x)$, the overall length of layers is is called $depth$ of model.

During the processing of neural networking training, we drive $f(x;\theta)$ to match $f^*(x)$ via produce a valure $\theta$ which make $y \approx f(x;\theta)\approx f^*(x)$

**Idea:** Choose $\theta$ so that $Hypothesis\ Function$ $\phi$ is close to $y$, ($\phi$ is denoted as $h_{\theta}$ in Andrew Ng's Course.).

### Gradient-Based Learning



The most difference between others linear models and neural network is that the *nonlinearity* of neural network causes most interesting loos functions to become **non-convex**.

Which is means that: *Stochastic gradient descent applied to **non-convex** loose function has **no convergence guarantiee**, And it's **sensitive** to the values of the **intial parameters**.*

So, It is important to initialize all weights to small random values, and the bias may be initialized to zero or to small positive values.

### Cost Function

In most cases, our parametric model defines a distribution $p(y | y;\theta)$ and we can simply using the principle of $maximum\ likihood$.

Most modern neural networks are trained using $maximum\ likelihood$. The $maximum\ likelihood$ is equivaliently describeed as the $corss-entropy$ between the training data and the model distribution.

The cost function is given by:


\begin{equation}
J(\theta)=- \mathbb{E}_{x,y~\hat{p}data}logP_{model}(y|x)
\end{equation}

For the case we using $MSE$ as cost function:

\begin{equation}
J(\theta)=\frac{1}{2}\mathbb{E}_{x,y~\hat{p}data}||y-f(x;\theta)||^2+const
\end{equation}


An adventange of using MLE as cost function is that it removes the burden of designing cost function for each model and for a speciy model $p(y | x)$, it automatically through out a cost funtion $log p(y|x)$.

### Output Units[3]

The choice of cost function is tightly coupled with the choice of output unit. Most of time, we simply use cross-entropy between the data distribution and the model destribution. Any kind of neural network unit that may be used as an output unit can also be used as *hidden unit*.

We suppose that the feedforwad network provide a set of hidden features defined by $h=f(x;\theta)$

* Linear units for Gaussian Output Distribution

A simple kind of output unit is based on an $affine\ transformation$ with **no nonlineary** which is often just called as linear units.

Given features $h$, a layer of linear output units producs a vector $\hat{y}=W^T+b$, It often used to produce the mean  of a conditional Gaussian distribution $p(y\ |x) = \mathcal{H}(y;\hat{y}, I)$

And **Maximizing the log-likelihood** is equivalent to *minimizing* the **Mean Squred error**.

* Sigmoid Units for Bernoulli Output Distributions

if a task is a binary classification problem, the maximum-likelihood approach is to define a $Bernoulli \ distribution$ over $y$ conditioned on $x$.

A $Bernoulli\ distribution$ is difined by just single number. The ANN needs to prodict only $P(y=1|x)$.

* Softmax Units for Multinoulli Ouput Distribution

Any time we wish to represent a probability distribution over a discrete variable with $n$ possible values, we may use the $softmax$ function. This can be seen as a generalization of the sigmoid function which was used to represent a probability distribution over a binary variable.

### Hidden Units

Note That: *The design of hidden units is an extremely active area of research and dose no yet have many definitive guiding theoretical principles.*

**Rectified linear units (ReLU)** is an extremely default choice of hidden unit.

### Architecture Design

Most neural networks are organized into groups of units called layers, cant arrange them in chain structure.

The first Layer is given by:

\begin{equation}
h^{(1)}=g^{(1)}(W^{(1)T}x+b^{(1)}
\end{equation}

And the second is given by:
\begin{equation}
h^{(2)}=g^{(2)}(W^{(2)T}h^{(1)}+b^{(2)}
\end{equation}


In chain-based architectures, the main architecture considerations are to choose the depth of the network and the width of each layer.

Deeper networks often are able to use far fewer units per layer and parameters, and often generlize to the test set, but are also harder to optimized.

### Back-Propagation

The $back-propagation\ algorithm$, open simply called backprop, allows the information from the cost to then flow backwards through the network, in order to compute the gradient.

The term *back-propagation* is often misunderstood as meaning the whole learning algorithm for multi-layer neural networks. Acutually, back-propagation refers only to the method for computing the gradient, while other algorithm, such as stochastic gradient decent, is used to perform learning using this gradient.

Futhermore, back-propagation is foten misunderstood as being specific to multilayer neural networks, but in principle, it can compute derivations of any function.

We will discribe how to compute the gradient $\Delta_xf(x,y)$ for an arbitrary function $f$, where x is a set of variable whose derviatives are desired, and $y$ is an additional set of variable that are inputs to the function whose derivations are not required.

#### Chain Rule of Calculus

Suppose that:

\begin{equation}
x\in \mathbb{R}\\
y=g(x)\ and\ z=f(g(x))=f(y)
\end{equation}
then the chain rule states that
\begin{equation}
\frac{\partial{z}}{\partial{x_i}} =\frac{\partial{z}}{\partial{y_i}}\frac{\partial{y_j}}{\partial{x_i}}
\end{equation}


We gan gerneralize this beyound the scalar case.

Suppose that:
\begin{equation}
x\in \mathbb{R}^m, y\in \mathbb{R}^n,\ g::\mathbb{R}^m \rightarrow \mathbb{R}^n
\end{equation}
and
\begin{equation}
f :: \mathbb{R}^n \rightarrow \mathbb{R}^m
\end{equation}
if
\begin{equation}
y=g(x)\ and\ z=f(y)
\end{equation}

then:
\begin{equation}
\frac{\partial{z}}{\partial{x_i}} = \sum_j{\frac{\partial{z}}{\partial{y_i}}\frac{\partial{y_j}}{\partial{x_i}}}
\end{equation}
In vector natation[5], this may be equivalently written as:
\begin{equation}
\nabla_xz=\left(\frac{\partial{y}}{\partial{x}}\right)^T\nabla_yz
\end{equation}

Where $\frac{\partial{y}}{\partial{z}}$ is the $n \times m$ Jacobian matrix[4] of $g$.

From this, we see that the gradient of a variable $x$ can be obetained by multiplying a Jacobian matrix $\frac{\partial{y}}{\partial{x}}$ by a gradient $\nabla_y z_1$, the back-proagation algorithm consists of performing such a jacobian-gradient product for each operation in the graph[6].

We use $\nabla_{\textbf{X}}z$ to denote the gradient of a value $z$ with tensor $\textbf{X}$. if $\textbf{Y}=g(\textbf{X})$, and $z=f(\textbf{Y})$, then:

\begin{equation}
\nabla_{\textbf{X}}z=\sum_j{(\nabla_{\textbf{X}}\textbf{Y}_j)\frac{\partial{z}}{\partial{\textbf{Y}_j}}}
\end{equation}


## Reference: 

[1][6] Book Deep Learning, Author Ian Gooodfellow and Yoshua Bengio and Aaron Courville, MIT Press

[2] Machine Learning Course, Andrew Ng. https://www.coursera.org/learn/machine-learning/

[3] Book Deep Learning, Author Ian Gooodfellow and Yoshua Bengio and Aaron Courville, MIT Press, page 190

[4] Jacobian Matrix https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant

[5] Vector notation https://en.wikipedia.org/wiki/Vector_notation



```python

```
