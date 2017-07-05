
<h1><center>SVM and Kernels</center></h1>
<center><h4>Ryan J. Kung</h4></center>
<center><h4>ryankung(at)ieee.org<h4></center>



## SVM


* Logistic Regression:

\begin{align*}
J(\theta) &= -\underset{\theta}{min} \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]+ \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2\\
&= \underset{\theta}{min}\frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}-\log (h_\theta (x^{(i)})) + (1 - y^{(i)})(-\log (1 - h_\theta(x^{(i)})))]+ \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2
\end{align*}

* Support Vector Machine
\begin{align*}
&= \underset{\theta}{min}\frac{1}{m} \sum_{i=1}^m [y^{(i)}cost_1(\theta^Tx^{(i)})) + (1 - y^{(i)})cost_2(\theta^Tx^{(i)}))]+ \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2\\
&= \underset{\theta}{min} \sum_{i=1}^m [y^{(i)}cost_1(\theta^Tx^{(i)})) + (1 - y^{(i)})cost_2(\theta^Tx^{(i)}))]+ \frac{\lambda}{2}\sum_{j=1}^n \theta_j^2\\
&= \underset{\theta}{min}C \sum_{i=1}^m [y^{(i)}cost_1(\theta^Tx^{(i)})) + (1 - y^{(i)})cost_2(\theta^Tx^{(i)}))]+ \frac{1}{2}\sum_{j=1}^n \theta_j^2 \ ;\ for\ C=\frac{1}{\lambda}\\
\end{align*}

* SVM Hypothesis:
\begin{align*}
h_{\theta}=\left\{
\begin{array}{0r}
1\ if\ \theta^Tx \geq 0\\
0\ otherwise
\end{array}
\right\}
\end{align*}




* Training Process:

if $y=1$, we want $\theta^Tx\geq1$

if $y=0$, we want $\theta^Tx\leq-1$

*This builds in an **extra safety factor or safety margin factor** into the support vector machine*

## Kernels

Q: Is there a different / better choice of features $f_1, f_2, f_3 ...$?

S: Given $x$, compute new feature depending on proximity to landmarks $l^{(1)}, l^{(2)}, l^{(3}$

\begin{align*}
f_i=similarity(x, l^{(i)})=exp(-\frac{||x-l^{(1)}||^2}{2\sigma^2})
\end{align*}

The mathematical term of the similarity function is **kernel** function, the specific kernel above is acutually called a *Gaussian Kernel*.

\begin{align*}
f_i=K(x, l^{(i)})=exp(-\frac{||x-l^{(1)}||^2}{2\sigma^2})
\end{align*}

If $x \approx l^{(i)}$: 

\begin{align*}
f_i \approx exp(-\frac{0^2}{2\sigma^2})\approx 1 
\end{align*}

If $x$ far from $l^{(i)}$: 

\begin{align*}
f_i=exp(-\frac{large\ number^2}{2\sigma^2})\approx 0
\end{align*}


### SVM with Kernels:

Given $(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}),..,(x^{(m)}, y^{(m)})$

Choose $(l^{(1)}, l^{(1)}), (l^{(2)}, l^{(2)}),..,(l^{(m)}, l^{(m)})$

Given example $x$: $f_i=K(x,l^{(i)})$

\begin{align*}
x^{(i)}\rightarrow \begin{bmatrix}f_1^{(i)} \newline f_2^{(i)} \newline \vdots \newline f_m^{(i)}\end{bmatrix}
\end{align*}

Hypothesis: Given $x$, compute fetures $f\in \mathbb{R}^{m+1}$, Predict "y=1" if $\theta^Tf\geq0$

Training: 
\begin{align*}
\underset{\theta}{min}C \sum_{i=1}^m [y^{(i)}cost_1(\theta^Tf^{(i)})) + (1 - y^{(i)})cost_2(\theta^Tf^{(i)}))]+ \frac{1}{2}\sum_{j=1}^{n=m} \theta_j^2
\end{align*}


Note that: $=\sum_{j=1}^{n=m} \theta_j^2=\theta^T\theta=||\theta||^2$ if we ignore $\theta_0$


SVM parameters:

$C(=\frac{1}{\lambda})$

* Large C: Lower bias, high variance,

* Small C: Higher bias, low variance.

$\sigma^2$

* Large $\sigma^2$: Feature $f_i$, vary more smoothly. Higher bias, lower variance.

* Small $\sigma^2$: Feature $f_2$, vary less smoothly. Lower bias, Higher variance.


```python

```
