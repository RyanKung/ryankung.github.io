<h1><center>Note about FRP and FHM</center></h1>

<center>**Ryan J. Kung**</center>
<center>**ryankung(at)ieee.org**</center>

## I Functional hybrid modeling

### 1.1 causal and none-causal languages

Special *modeling languages* have been developed to facilitate modeling and simulation. There are two broad language categories in this domain. 

**Causal (or block-oriented) languages** and **none-causal (or object-oriented) language**.

In causal modeling, the equations that represent the physics of system must be written so that the direction of singal flow, in $causality$, is explicit.

*E.g:* 

* [Simulink (Mathlab)](https://www.mathworks.com/products/simulink.html)[2]
* [Ptolemy II](http://ptolemy.eecs.berkeley.edu/ptolemyII/)[3]

	Ptolemy II is an open-source software framework supporting experimentation with **actor-oriented design**. [4]

"In Non-causal language, the equations taht focuses on the interconnection of the components of system being modeled, from which causality is then inferred."

*E.g:*

* [Dymola (Mathlab)](https://www.mathworks.com/products/connections/product_detail/product_35341.html)[5]
* Modelica[6]

#### 1.1.1 Drawbacks:

* causal languages:  Needing to explicity specify the causality. This hampers modularity and reuse[7].

* none-causal languages: Trid to solve the issue of cause languages via loowing the user to avoid committing the model itself to a specific causality. But current non-causal modeling languages sacifice generality, particularly when it comes to hybrid modeling.

* Additional weaknesses:  Language safety disciplines are uncommon.


### 1.2 Functional reactive programming

In research of Yale, they has developed a framework called *functional rective programming*, or FRP[8]. which is highly suited fro causal hybird modeling[9]. And, because the full power of a functional language is avaliable, it exhibits a high degree of modularity, allowing reuse of components and design patterns.[10]


### 1.3 Functional hybrid modeling

*functional hybird modeling*, or FHM is a combined of FRP and none-causal languages. Which can be seen as a generalization of FRP, since FRP's functions on singals are a special case of FHM's relations on signals. FHM, like FRP, also allows the description of structurally dynamic models.[11]

## II Integrating Functional Programming and Non-Causal Modeling[12]

*The two key idea are to give first -class status to relations on signals and to provide constructs for discrete switch ing between relations.*

### 2.1 First-Class Signal Relations

A *signal* is, conceptually, a function of time. A *signal function* maps a stimulating singal onto a responding signal. A natual mathematical description of a continuous signal function is ahta of an ODE(ordinary differential equation) in explict form.

A *function* is just a special case of the more general concept of a *relation*. While functions usually are given a causal interpretation, relations are inherently non-causal. DAEs(differential and algebraic equations) wre at the heaart of none-causal modeling. It's natureal to view ODE in explict form can be seen as a causal signal function.

A non-causal model is an implicit system of DAE: $f(x, x',w,u,t)=0$,where $x$ is a vector of state variables, $w$ as a vector of algebraic variables, $u$ is a vector of inputs, and $t$ is the time.

Conceptualy, we define the polymorphic type of signal as $S\ \alpha = Time \rightarrow \alpha$; that is, $S\ \alpha$ is the type of a signal whose instantaneous value is of type $\alpha$(parameteric type).



## Reference

[1] [Andrew Kennedy. Programming Languages and Dimensions. PhdD thesis, University of Cambridge, Computer Laboratory, April 1996. Published as Technical Port No. 391.](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-391.pdf)

[2] Simulink: https://www.mathworks.com/products/simulink.html

[3][4] Ptolemy Project: Ptolemy II http://ptolemy.eecs.berkeley.edu/ptolemyII/

[5] Dynamic Modeling Laboratory https://www.mathworks.com/products/connections/product_detail/product_35341.html

[6] Modelica  https://www.modelica.org/

[7] Frannois E. Cellier. Object-oriented modelling: Means of dealing with system complexity. In Proceeedings of the 15th Benelur Meeting on Systems and Control, Mierlo, The Netherland, pages 53-64, 1006 cited in  Functional Hybird Modeling, Henrik Nilsson, John Peterson, and Paul Hudak, Department of Computer Science, Yale University, PADL 2003

[8] Zhanyong Wan and Paul Hudak. Functional reactive programming from first princple. In proceeding s of PLDI'01: Symposium on Programming Language Design and Implementation, pages 202-202, June 2000.

[9][10][11] Henrik Nilsson, John Peterson, and Paul Hudak, Functional Hybird Modeling, Department of Computer Science, Yale University, PADL 2003

[12][13] Henrik Nilsson, John Peterson, and Paul Hudak, Functional Hybird Modeling, Department of Computer Science, Yale University, PADL 2003
