
<h1><center>Neural Networks: Convlutional and pooling</center></h1>

<center><h4>Ryan J. Kung</h4></center>
<center><h4>ryankung(at)ieee.org<h4></center>

CNNs are a specialized kind of neural network for parocessing data that has a known, grid-like topology. **Convolutional networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers**[1].


## The Convolution Operation

Convolution is an mathematical operation on two real-valued functions, $f, g$[2].

\begin{equation}
(f * g)(t) = \int_{-\infty}^{\infty}{f(\tau)g(t-\tau)d\tau}
\end{equation}

The discrete convolution can be defined as:

\begin{equation}
(f * g)(t) = \sum_{\tau=-\infty}^{\infty}{f(\tau)g(t-\tau)d\tau}
\end{equation}

In two-dimensional case, We have two-dimensional data $I$ as input, and using a two-dimensional kernel $K$, the shape of $I, K$ is $(m, n)$:

\begin{equation}
(I * K)(i, j) = \sum_m \sum_n I(m, n)K(i-m, j-n)
\end{equation}

Convolution is commutative:

\begin{equation}
(I * K)(i, j) = \sum_m \sum_n K(i-m, j-n)I(m, n)
\end{equation}

The commutative property of convolution arises because we have $flipped$ the kernel relative to the input, in the sense that as m increased, the index into the input increases, but the index into the kernel decreases.

Many ML librarys implement a realted function called $cross-correlation$, which is the same as convolution but without flipping the kernel, and call it `convolution`:

\begin{equation}
(I * K)(i, j) = \sum_m \sum_n I(i-m, j-n)K(m, n)
\end{equation}

We usually call both operations convolution and specify whether flipped the kernel. **An algorithm based on convolution with kernel flipping will learn a kernel that is flipped relative to the kernel learned by an algorithm without the flipping.**

Discrete convolution can be viewd as multiplication by a matrix. However, the matrix has several entries constrained to be equal to other entries. For univariate discrete convolution, each row of the matrix is constrained to be equal to the row above shifted by one element, which is known as a $Toeplitz\ matrix$, and in two dimensions, it's a $boubly \ block\ circulant\ matrix$, responsed to convolution[3].

Convolution usually corresponds to a very sparse matrix, this during to the kernel is usually much smaller than the input image. **Any neural network algorithm that works with matrix multiplication and doses not depend on specific properties of the matrix structure should work with convolution, without requiring any futher chances to the NN.[4]**

## Motivation



Convolution leverages three ideas that can help improve a machine learning system: **sparse interactions**, **parameter sharing** and **equivariant representations**.

#### Sparse Interactions

Instead matrix multiplication used by traditional neural network, Convolutional networks have $sparse\ interations$ (alwo referred to as $sparse\ connectivity$ or sparse weights), which is accomplished by making the kernel smaller than the input.

If we keep $m$ seral orders of magnitude smaller than $m$, we can see quite large improvements.

#### Parameter Sharing

$Parameter\ sharing$ refers to using the same parameter for more than one function in a model. In a traditional NN, each element of the weight matrix is used exactly ones. As a synonym for parameter sharing, one can say that a network has $tied\ weights$, because the value of weight applied to one input is tied to the value of weight applied elseware.[5]

In CNN, each member of the kernel is used at every position of the input. The parmeter sharing used by ANN is means that, **rather than learning a separate set of parameters of every location, we learn only one set**. This does not affect the runtime of forward progagation -- it's still $O(k\times n)$-- but it does futher reduce the storage requirements of model to $K$ parameters which is usually several orders of magnitude less than $m$.

With $Sparse\ interactions$ and $Parmeter\ Sharing$, convolution is thus dramatically more efficient than dense matrix multiplication in terms of the memory requirements and statistical efficiency.

#### equivariant representations[6]

The *particualar form* of *parameter sharing* causes the layer to have a property called $equivariance$ to translation. to say a function is equivariant means that if the input changes, the ouput changes in the same way. For example:

$f(x)$ is equivariant to $g$ $iff$: $f\circ g(x) = g \circ f(x)$

In the case of convolution, if $g$ is any function that translates the input(i.e, shift it), then the convolution function is equivariant to $g$.

In the case of processing **time series data**, this means that convolution produces a sort of timeline that shows when different features appear in the input. If we move an event later in time in the input, the exact same representation of it will appear in the output, just later in time.

With the case of processing 2-D images,  convolution creates a 2-D map of where certain features apear in th input. If we move the object in the input, its representation will move the same amount in the output.

### Pooling

A typical layer of CNN include three stages. In the first state, the layer performs several convolutions in parallel to produce a set of linear activations. Then in the second state, each linear *activation* is run through a nonlearn activation function, such as *ReLU*, this stage is called the $decector\ state$. In the third stage, we use a $pooling\ function$ to modify the output of the layer further.

A pooling function replaces the output of the net at a certain location with a summary statistic of the nearby outputs. For example, the $max\ pooling$ operation reports the maximum output within a rectangular neighborhood.

In all cases, pooling helps to make the representation become approximately $invariant$ to small translation of input. **Invariance to local translation can be a vary useful property if we care more about whether some feature is present than exactly where it is.**[7]

The use of pooling can ve viewed as adding an infinitely strong prior that the function the layer learns must be **invariant** to small translation. When this assumption is correct, it can greatly improve the statistical efficiency of the network.

Because pooling summarize the response over a whole neightborhood, it's possible to use fewer pooling units than the dectector units,  by reporting summary statistics for pooling regions spaced $k$ pixels apart rather than 1 pixel apart.

[1][4][5][6][7] Book Deep Learning, Author Ian Gooodfellow and Yoshua Bengio and Aaron Courville, MIT Press, page 330

[2][3] Convolution, wikipedia https://en.wikipedia.org/wiki/Convolution


```python

```
