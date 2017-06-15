<h1><center>Metaphysical Micro-Services Modeling</center></h1>

<center>**Ryan J. Kung**</center>
<center>**ryankung(at)ieee.org**</center>
## I Metaphysical Modeling

There are two ways in which the term modeling can be understood:
**descriptive** and **preciptive**. A descriptive model represent an existing
system Thus a presciptive model is one that can be used to construct the target
system[1].

### 1.1 Descriptive with Functional

A Functional is a Function which accept a paramater $\theta$ and output a
function $f(x;\theta)$, where $f(x)$ is the abstract of $services\ processing$.

#### 1.1.1 *Pure Function* and *equivariant*

Pure Function is that a function dose not cause any side-effect, which is
means, no statements, IO, wont cause *MUTEX*, etc.

And suppose that there is two function $f$ and $g$, $f(x)$ is equivariant to
$g$ $iff$: $f\circ g(x)=g \circ f(x)$.

#### 1.1.2 Compose Function

In mathematic, compose of functions $f(g(x)) = f \circ g (x)$ can
can be represent that $compose ( f, g)(x)$, or in S-exp $(compose\ f'\ g')$,
where $\circ$ is a $left\ prefix$ function.

Functions can be composed with any order iff they are $Pure$ or $Equivariant$, otherwise we may need to define some fixed oreder function with $J=f\circ g$,
which have some hidden functions like $f$ and $g$ in our descrpiptive scope.


#### 1.1.2 Side-Effect

Side-effect usually caused via *Resource Sharing*, *IO*, or *Statement*. If a
function cause side-effect, it's obviously not a $pure$ function, but it still
can be describe as $equivariant$ function.

A $distribution$ system can be describe as *a system which including
unpredictable order in which certain events can occour*, So as the
micro-service System. With the definition of the $Lamport\ Timestamp$, we can
define a compose of functions like: $Ev_a \rightarrow [Ev_{b1}, Ev_{b2}]
\rightarrow Ev_c \rightarrow Ev_d$. If all events are $pure$ or $equivariant$,
we can think $Ev_{b1}$ and $Ev_{b2}$ are occour concurrency. 

But if $Ev_{b2}$ couse side-effect, we must include the statement of $Ev_{b2}$
in our $Lamprot\ Timeline$: $Ev_a \rightarrow [Ev_{b1}, Ev_{b2}^1] \rightarrow
[Ev_{b1}, Ev_{b2}^2]\rightarrow Ev_c \rightarrow Ev_d$. The processing of
$[Ev_{b1}, Ev_{b2}^1] \rightarrow [Ev_{b1}, Ev_{b2}^2]$ is a block or
synchronization of side-effect.

### 1.2 Preciptive with DSLs

When we said that a language $l$ covers a subset of $P$, we can simply call
this subset the domain covered with $l$. The subset of $P$ in that domain $P_D$
is equal to the subset of $P$ we can express with a language $l$ $P_l$. So, we
cannot ask a question like: “Does the language adequately cover the domain?”,
since it always does, by definition.[4] And the definition can be also interept
as that “DSL is always model complete, by definition”.

DSLs is actually a meta-ness of preciptive modeling, and according to the
*Model Complete Theory*, DSLs is the shortcut of providing a better abstrict of
target system. Or we can say that, all modeling methods will bring us a DSL
which is model completion.

### 1.2 Scale, Underfitting and Overfitting

We can optimize $f(x;\theta)$ by provide a better scale rate $\theta$. So we
may needs to define a $Cost\ function$ to define how good a choice of $\theta$
is, this $cost\ function$ is always calls banchmark in architecture modeling.

For avoid underfitting and overfitting, we usually setup some monitor services
for getting the approximate value of $\theta$.

## II Concrete

Thus we have passed the age of exploration of the distribution system or
archicture. There is alot of components for building a concrete distribute
micro-services system.

### 2.1 Describe the Services

In a distribute system envirement, we can build out system on a implemented
distribute system, such as `etcd`, or `consul` Which provide an opensource
implementation of  `raft` algorthm that equivariant to Lamport's PAXOS.

The best way for descriptive modeling should be DSLs, thus it's hard to
implementation. In a compromise way, we can use `RPC` to provide a better
description of services itself. `GRPC` might be a good choice, during to the
strict type system and `Http 2.0` supporting.

A services should tell others *Who* it is, and *What* or *How* it should be.
For the formalize denote of $y=f(x;\theta)$, $f$ should be the `id` or `name`
of the services, and $x\rightarrow y$ is the input/output protocol of services.
For more, It should also provide $\theta$ to other services which is usuallys
stands for $scale\ rate$. So it's necessary to **register** the meta
information as `K/V` with our distribute system for **exploring** by other
services. And use services like `confd` to moniting and applying the `KV`
changes.

For static config of services(even for clients), we can simple build a static
config system. For example, `PSS` is a simple K\V storage system base on
`pandas` which provide map, reduce, filter function for reshape/recache the
static config files.

### 2.2 Regulator of $\theta$

For getting a better $\theta$, We may need some monitor services such as
`grafana / Prometheus / Sentry` for function of cost. Then we may try to
predict $\theta$ based on the banchmark and result of montoring.

### 2.3 Side-effect

It's really complex for managing statements and side-effect, visiting a resoure
by multiple services in same time may occor unpredictable result or
$mutex\rightarrow deadlock$. To avoid it, we may needs to discript the
side-effect more strictly. Some GPL such like `Haskell` provide a abstruct data
structure for manage it called `monad`, thus there is `IO monad`, `State
Monad`, `Pal Monad`, `Eval Monad`.

`Qubit`[4] is thus a system for manage side-effect data. The basic idea of
Qubit is from `FRP` and `Eval Monad`. Qubit is a multi-process services which
is also distribute but has no effect. A `Qubit` Works as a `Immutable Type
Class` which is just provide some eval monad for schedular and calling. The
Qubit descriped with following rules.

* Eval Monad
* Safe DSL :: Require('xxx.py')
* Data -> [prevData, NextData]
* Save $iff$ side effect
* Timestamp and Lamport-Timestamp
    $Ev_1 \rightarrow [Ev_2, Ev_3] \rightarrow Ev_n$ 

### 2.4 Toolkits

For depoly and testing the distribute system, we may need alot of tookits. Such
as `git-daemon` or `fabric`. `Stack` is another choice for making building
system easilyer which provide a remote shell based on websockets, and support
fabric's systax which is actually a DSL


# III Reference 

[1][4] DSL Engineering, Designing, Implementing and Using Domain-Specific
Languages, Markus Voelter, dslbook.org, ch 2.1

[2]Time, Clock, And the Ordering of Events in a distribute system, Leslie
Lamport, MCA Inc.

[3] Why DSL Ryan J. Kung,
https://ryankung.github.io/posts/2017-06-14-why-dsl.html

[4] Qubit, A Eval based Time Series Event System.
https://github.com/RyanKung/qubitcal Micro-Services Modeling