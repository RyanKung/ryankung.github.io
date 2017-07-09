
<h1><center>Note of Unsupervised Learning: K-Means and PCA</center></h1>


<center>**Ryan J. Kung**</center>
<center>**ryankung(at)ieee.org**</center>

## I K-means


### 1.1 optimization object

$c^{(i)}=$index of cluster $i\ |\ i\in[1,k],i\in\mathbb{R}$, to which examle $x^{(i)}$ is currently assigned

$\mu_k$= cluster centroid  $k\ (\mu_k\in\mathbb{R}^n)$

$\mu_{c^{(i)}}$ = cluster centroid of cluster to which example $x^{(i)}$ has been assigned.


Optimization Objetive (distorting function):

\begin{align*}
J(c^{i},...,c^{(m)}, \mu_1,...,\mu_k)=\frac{1}{m}\sum_{i=1}^m||x^{(i)}-\mu_{c^{(i)}}||^2
\end{align*}

Goal:

\begin{align*}
\underset{(c^{i},...,c^{(m)},\\ \mu_1,...,\mu_k)}{min}J(c^{i},...,c^{(m)}, \mu_1,...,\mu_k)=\frac{1}{m}\sum_{i=1}^m||x^{(i)}-\mu_{c^{(i)}}||^2
\end{align*}





K-means algorithm:

Randomly initalize $K$ cluster centroids $\mu_!, \mu_2, ..,\mu_k \in \mathbb{R}^n$.


Repeat {

......for i=1 to m (**cluster assignment step**)

.........$c^{(i)}$:=index {from 1 to $K$) of cluster cntroid closest to $x^{(i)}$
        
......for $k=1$ to $K$ (**move centroid step**)
    
.........$\mu_k$:=mean of points assigned to cluster $k$

}

The **cluster assignment step** is actually does:
\begin{align*}
\underset{(c^{i},...,c^{(m)},\\ with\ holding\ \mu_1,...,\mu_k)}{min}J(...)\ wat\ c^{(i)}\ |\ i\in[1,k]
\end{align*}


The **move centroid step** is actuall does:
    
\begin{align*}
\underset{( with\ holding\ c^{i},...,c^{(m)},\\\mu_1,...,\mu_k)}{min}J(...) wat\ \mu_i \ | \ i \in [1, k]
\end{align*}
   

### 1.2 Random initazation

Randomly inialize $K$ cluster centroids $\mu_i\ |\ i\in[1,K];\mu_K\in \mathbb{R}^n$

0) Should have $K<m$

1) Randomly pick $K$ traing examples

2) set $\mu_1,...,\mu_K$ equal to these $K$ examples.

For solving **Local otima Problem**, you may needs to try multi-time **random initazation**.





* Concretely:

For i=1 to 100 {

......Randomly initialize K-means.

......Run K-means; Get $c^{(i)},\mu_j\ |\ i\in[1,m]; j\in[1,K];(i,k)\in(\mathbb{R},\mathbb{R})$

......Compute cost function (distortion) $J(c^{i},...,c^{(m)}, \mu_1,...,\mu_k)$

}

Pick clustering that gave lowest cost $J(.)$

if $K\in [2, 10]$, by doing multiple random inializations can ofen provide a better local optima.

But if $K$ is very large, $K\geq10$, then havling multiple random initializations is less likely to make a huge difference and higher change taht your fist random initialization will gave you a pretty decent solution already.

#### 1.2.1 Chooing the value of K: The Elbow method



Elbow method: 
\begin{align*}
\underset{J(c^{i},...,c^{(m)}, \mu_1,...,\mu_k;K)}{Elbow}
\end{align*}

The known issue is, the Elbow method may not provide a clear answer, but it's worth for trying. Another way to Evaluate K-means is base on a metric for how well it performs for that later purpose.

## II Dimensionality Reduction

* Motivation I: Data Compression:

* Motivation II: Visualization


### 2.1 Principal Component Analysis (PCA)

The goal of PCA is tried to Reduce from n-dimension to k-dimension: Find $k$ vector $u^{(i)}\in\mathbb{R}^n;i\in[1,k]$, that the sum of squares of $x^{(i)}$ is minimized which is sometimes also called the **projection error**.

#### 2.1.1 Data processing:

Training set: $x^i\ |\ i\in[1,m]; i\in\mathbb{R}$

Processing (feature scaling/mean normalization):
\begin{align*}
\mu_j=\frac{1}{m}\sum_{i=1}^mx_{j}^{(i)}
\end{align*}

Replace each $x_j^{(i)}$ with $x_j-\mu_j$

if different features on fifferent scales, scale features to have comparable range of values.

\begin{align*}
x_j^{(i)}=\frac{x_j^{(i)}-\mu_j}{s_j}
\end{align*}



#### 2.1.2 PCA

Goal: Reduce data from $n$-dimensions to $k$-dimensions:

Compute `Covariance Matrix` (which is always satisfied *symmetic positive definied*):

\begin{align*}
\Sigma=\frac{1}{m}\sum_{i=1}^n(x^{(i)})(x^{(i)})^T
\end{align*}

Compute `eigenvectors` of meatrix $\Sigma$:

\begin{align*}
[U, S, V] = svd(\Sigma)\ |\ \Sigma\in\mathbb{R}^{nxn}
\end{align*}

svd: **Sigular value decomposition**


\begin{align*}
U=\left[
\begin{matrix}
\vdots&\vdots&\cdots&\vdots&\cdots&\vdots\\
u^{(1)}&u^{(2)}&\cdots&u^{(k)}&\cdots&u^{(n)}&\\
\vdots&\vdots&\cdots&\vdots&\cdots&\vdots\\
\end{matrix}
\right]
\in\mathbb{R}^{nxn}
\end{align*}

\begin{align*}
U_{reduce}=\left[
\begin{matrix}
\vdots&\vdots&\cdots&\vdots\\
u^{(1)}&u^{(2)}&\cdots&u^{(k)}\\
\vdots&\vdots&\cdots&\vdots\\
\end{matrix}
\right]
\in\mathbb{R}^{nxk}
\end{align*}

Thus:

\begin{align*}
z={U_{reduce}}^T x\ |\ x\in\mathbb{R}^{nx1};Z\in\mathbb{R}^{kx1}\\
\end{align*}

So mapping With PCA:

\begin{align*}
z^{(i)}=U_{reduce}^Tx^{i}=({u_{reduce}}^{(i)})^Tx
\end{align*}



With `mathlab`:


```mathlab
Sigma = (1/m)*X'*X;
[U, S, V] = svd(Sigma);
Ureduce = U(:, 1: k);
z = Ureduce'*x'
```

#### 2.2.3 Chooing the value of K (Number of Principle Components)


* Minilize Averge squared project error:

\begin{align*}
\frac{1}{m}\sum_{i=1}^m||x^{(i)}-x_{approx}^{(i)}||^2
\end{align*}
*  Total variation in the data:
\begin{align*}
\frac{1}{m}\sum_{i=1}^m||x^{(i)}||^2
\end{align*}


* Typically, choose $k$ to be smallest value so that:
\begin{align*}
\frac{\frac{1}{m}\sum_{i=1}^m||x^{(i)}-x_{approx}^{(i)}||^2
}{\frac{1}{m}\sum_{i=1}^m||x^{(i)}||^2}\leq0.01
\end{align*}

Thus we can say "99% of variance is retained"

#### Algorithm:

Try PCA with $k=n$

Compute:
\begin{align*}
U_{reduce},Z^{(i)},x_{approx}^{(i)}\ |\ i\in[1,m]
\end{align*}

Check if:

\begin{align*}
\frac{\frac{1}{m}\sum_{i=1}^m||x^{(i)}-x_{approx}^{(i)}||^2
}{\frac{1}{m}\sum_{i=1}^m||x^{(i)}||^2}\leq0.01
\end{align*}


**For Optimize**:


\begin{gather}
[U, S, V] = svd(\Sigma)
\end{gather}

Thus:

\begin{gather}
S=\begin{bmatrix}
s_{11}\\
&s_{22}& & {\huge{0}}\\
&&s_{33}\\
& {\huge{0}} && \ddots\\
&&&& s_{nn}
\end{bmatrix}
\in\mathbb{R}^{nxk}\end{gather}


For given $K$:

\begin{align*}
\frac{\frac{1}{m}\sum_{i=1}^m||x^{(i)}-x_{approx}^{(i)}||^2
}{\frac{1}{m}\sum_{i=1}^m||x^{(i)}||^2}\leq0.01
\end{align*}

Can be compute by:

\begin{align*}
1-\frac{\sum_{i=1}^kS_{ii}}{\sum_{i=1}^nS_{ii}}\leq0.01
\end{align*}


### 2.2 Reconstruction from Compressed Representation



\begin{align*}
z={U_{reduce}}^T \\
X_{approx}=U_{reduce}. z
\end{align*}


### 2.3 Application:

* Compression:
    - Supervised learning speedup
    - Reduce memory / disk needs to store data

* Visulization

#### 2.3.1 Bad use of PCA: To prevent overfitting:

Use $z^{(i)}$ instead of $x^{(i)}$ to reduce the number of features to $k<n$,

Thus, fewer features, less likely to overfit.

It might work OK, but **isn't a good way** to **address** overfitting. The fact of PCA does here, is throwed away some information.



## III Reference

[1] Machine Learning, Andrew Ng, https://www.coursera.org/learn/machine-learning


```python

```
