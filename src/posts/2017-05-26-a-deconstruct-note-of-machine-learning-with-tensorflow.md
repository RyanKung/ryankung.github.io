<h1><center>A Deconstruct Note of Machine Learning with Tensorflow</center></h1>

 

<center>**Ryan J. Kung**</center>
<center>**ryankung(at)ieee.org**</center>

 

<h4><center>Abstract</center></h4>

This note provide a inside vision of how tensorflow works on Machine Learning processing. We first described how MNIST DataSet implements and it's data structure, Then we used a classic linear machine method to create a MNIST DataSet based handwritting number recognize application. 


## I MNIST
 

MNIST DataSet, As the *hello world* of Minchine Learning, we usually download it from Yann LeCun's website (http://yann.lecun.com/exdb/mnist/), and in Tensorflow, MNIST can be easily fetched by follow codes.


The implementations can be checked via function `read_data_set` at [fn read_data_sets()](`https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py#L211`), which accept several parameters.

```python
def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None) -> DataSet:

```

For the case `fake_data == True`, it returns a Fake DataSet, *`DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)`*, otherwise, It will returns the MNIST dataset with a high dimension dataset `base.Datasets(train=train, validation=validation, test=test)`, which include training data:label, validation data:label, and test data:label.

```python

  options = dict(dtype=dtype, reshape=reshape, seed=seed)
  
  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)

  # The origin implementation is dirtier than this, there is a pull request
  # https://github.com/tensorflow/tensorflow/pull/10188

```

And it worth to notify that the `fake DataSet` is not actuall a `Empty DataSet`, the `fake_data` label parameters will cause `side-effect` of how `fn DataSet(fake_data=True).next_batch` works.

The above `one_hot` label means, the labels will be present as `one-hot vertaxs


\begin{equation}
label(2) := [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]\\ \\
label(1) := [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
\end{equation}

\begin{equation}
minst_i = (Set(Data(image_i)), Set(label_i))
\end{equation}

\begin{equation}
Data(image_i) = \sum_{r=0}^w\sum_{h=0}^h pixel(image_i, r, h)
\end{equation}


Lets check about $Data(image_i)$, You can use `matplotlib` to display some of them.

```python
import matplotlib as plt
plt.imshow(np.reshape(mnist.train.images[0], (28, 28)), cmap="gray")
```


```python
%matplotlib inline
import numpy as np
import matplotlib

fig = plt.figure(figsize=(80,80))
for i in range(1, 31):
    img = np.reshape(train_data.images[i-1], (28, 28))
    ax = fig.add_subplot(8, 10, i)
    matplotlib.pyplot.imshow(img, cmap="gray")
```


![png](/images/output_12_0.png)


All images are shuffled, and you can reshuffle them again by applying function `numpy.random.suffle` on vetaxs.

## II  Machine Learning with MNIST via Softmax Regressions


### 2.1 Linear Regression

Now we have the input image Vertax with 28x28 dimensionals, and needs a function which can map it into a 10 dimensionals Vertax as the one-hot Vertax label. All we needs is to create a Martix with shape (10, 28*28), and a bias. Bias is often used by Term $linear\ regression$ to refer to a slightly more sophisticated model[5]. Thus the function formula can be descibe as:


\begin{equation}
\left[
\begin{matrix}
y_11\\
y_12\\
\vdots\\
y_{10}
\end{matrix}
\right]=
\left[
\begin{matrix}
W_{1,1}&W_{1,2}&W_{1,3}&\cdots&W_{1,784}\\
W_{2,1}&W_{2,2}&W_{2,3}&\cdots&W_{1,784}\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
W_{10,1}&W_{10,2}&W_{10,3}&\cdots&W_{10,784}
\end{matrix}
\right] .
\left[
\begin{matrix}
x_1\\
x_2\\
\vdots\\
\vdots\\
\vdots\\
x_{783}\\
x_{784}
\end{matrix}
\right] +
\left[
\begin{matrix}
b_1\\
b_2\\
\vdots\\
b_{10}
\end{matrix}
\right] 
\end{equation}

Or:
    
\begin{equation}
f(x) = \sum_{i=1}^{10} w_{i,j}x_j+b_i=W^TX+B
\end{equation}


### 2.2 Multionoulli distribution

`Softmax function` can be seen as a generalization of the `sigmoid function`[1], which is also named as `logistic function`[8] and used to predict the probabilities associated with a `multionoulli distribution`.

\begin{equation}
softmax(x)_1 = \frac{exp(x_i)}{\sum_{j=1}^nexp(x_j)}
\end{equation}

`Multinoulli Distribution` also knowns as `categorical distribution`, which is a distribution over a single discrete variable with $k$ deifferent states, where $k$ is finite.[2] It's parametrized by a vector $P \in [0, 1]^{k-1}$, where $p_i$ gives the probability of the $i$-th state. In this case, $softmax$ will help us to build a `Multinoulli Distribution` with a $k=10$ `categorys`. 

### 2.3 Implementing with tensorflow

Initialize Vertaxs and Matrixs First:

```python
x = tf.placeholder(tf.float32, [None, 784])
```
$x$ is a placeholder of input images which have 28*28 dimensionals.

```python
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```
Matrix $W$ and Vertax $b$ are sets to all `Zero`.

Thus, We start to implement formule $f(x)=W^TX+B$ and applied with function $softmax$:

```python
logits = tf.matmul(x, W) + b
y = tf.nn.softmax(logits)
```


### 2.4 Training with Maximum Likelihood Estimation

Machine Leaning is that “$A$ computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$ ."by Mitchell(1997).

So, fot training, we needs to find out a method for $Performance$ measurement, which descript what is good or bad and iterate with `optimization` $Task$ and getter $Experience$ via dataset. Optimization refers to the "task of either minimizing or maximizing some function $f(x)$ by altering $x$".[4] We usually call the function we want to minimize or maximize as $cost\ function$, $loss\ function$ or $error\ function$.

The `Maximum likelihood principle` is the most common model for making a good estimater of training models. Consider we have training DataSet $\mathbb{X}={x_{(1)}, x_{(2)}, \cdots, x_{(n)}}
$, and a distribute mode $p_{model}(x;\theta)$ which is based on $\theta$. The Maximum Likelihood Estimator is defined as:


\begin{equation}
\theta_{ML} = argmax_{\theta} \ p_{model}(\mathbb{X};\theta)\\
=\mathop{argmax}_{\theta}^{} \prod_{i=1}^m\ p_{model}(x^{(i)};\theta)
\end{equation}

Where:
\begin{equation}
\mathbb{X}={x_{(1)}, x_{(2)}, \cdots, x_{(n)}}
\end{equation}

And we can simply and equivalent trans the product function with sum function:


\begin{equation}
\Leftrightarrow \mathop{argmax}_{\theta}^{} \sum_{i=1}^m\ \log p_{model}(x^{(i)};\theta)
\end{equation}

Ony way to interpret maxium likelihood estimation is to view it as minimizing the dissimilarity between the empirical distribution $P_{data}$ defined by the training set and the model distribution, weith the degreee of dissimilarity between the two measured by the $KL$ divergence.

\begin{equation}
D_{KL}(p_{data}||p_{model})-\mathbb{E}_{x~p_data}[log\ p_{model}(x)]
\end{equation}

Minimizing this KL divergence corresponds exactly to minimizing the `cross-entropy` between the distributions.[6]

To estimate a conditional probability $P(y|x;\theta)$, If $X$ presents all out inputs and $Y$ all out oversved targets, then the CML is:
\begin{equation}
\theta_{ML} = \mathop{argmax}_{\theta}^{} \sum_{i=1}^m\ \log P(y^{(i)}|x^{(i)};\theta)
\end{equation}

or 

\begin{equation}
H_{y'}(y) = -\sum_i y'logP(y_i))
\end{equation}

as the documents of tensorflow[4].

### 2.5 Cross-entropy traing with tensorflow

For cross entropy implementation with Tensorflow, it only needs two lins of code:
```python
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
```

Then we choosed Grandient Descent Optimizer[7] for minimize cross_entropy:
```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

### 2.6 Go Train

Before training start, you may needs to check your hardware resource by code 
```python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```

First launch the model session with GPU.
```python
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True)
session = tf.InteractiveSession(config=config)
```

Then we'll run the training step 1000 times.

```python
config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True)
sess = tf.InteractiveSession(config=config)
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

And Evaluating via below codes:
    
```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```


## Reference


[1][2][4][5][6] Book Deep Learning, Author Ian Gooodfellow and Yoshua Bengio and Aaron Courville, MIT Press, page 198

[3][4] MNIST For ML Beginners  https://www.tensorflow.org/get_started/mnist/beginners

[7]Machine Learing MOOC, Andrew Ng. https://www.coursera.org/learn/machine-learning/exam/wjqip/introduction
