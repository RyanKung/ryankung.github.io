
<h1><center>The Process Model of CSP, With Haskell<center></h1>
<center>**Ryan J. Kung**</center>
<center>**ryankung(at)ieee.org**</center>


## I Process

* Each event name denotes an event class; there may be many occurrences of events in a single class, separated in time.

* The set of names of events which are considered relevant for a particular description of an object is called it's $alphabet$. *It is locically impossible for an object to engage in an event outside it's alphabet*[1].

We use word $process$ to stand for the behaviour pattern of an object, thus we use the following conventions[2].

##### 1. Wrods in lower-case letters denote distinct events, e.g.

$coin$, $choc$, $in2p$, $out1p$, or $a$, $b$, $c$, $d$

##### 2. Words in upper-case letters denote specific defined processes, e.g

$VMS$ - the simple vending machine

$VMC$ - the complex vending machine

and the latters $P, Q, R$(occuring in laws) stands for arbitary processes.

##### 3. The letters  $x, y, z$ are variables denoting events.

##### 4. The latters A, B, C stand for sets of events

##### 5. The latters $X, Y$ are variables denoting processes.

##### 6. The alphabet of process $P$ is denotes $\alpha P$, e.g:

$\alpha VMS = \{coin, choc\}$

$\alpha VMC = \{in1p, in2p, small, large, out1p\}$

The **process** with alphabet $A$ which never actually engages in any of the events of $A$ is called $STOP_A$

### 1.1 Prefix Notation

Let $x$ be an event and let $P$ be a process, Then

$(x \rightarrow P)$ (proounced "$x$ then p")

describes an object which first engages in the event $x$, then behaves exactly as described by $P$. The process $(x \rightarrow P)$ is defined to have same alphabet as $P$, so this notation must not be used unless $x$ is in that alphabet; more formally:

$\alpha (x \rightarrow P) = \alpha P|\ x \in \alpha P$

**Examples**

* $(coin \rightarrow STOP_{\alpha VMS})$

* $(coin \rightarrow (chor \rightarrow (coin \rightarrow (choc \rightarrow STOP_{\alpha VMS}))))$

We can omit brackets in the case of linear sequences of events.


$(coin\rightarrow chor\rightarrow coin\rightarrow choc \rightarrow STOP_{\alpha VMS})$

### 1.2 Recursion

Condider the simplest possible everlasting object, a lock which never stop.

$\alpha CLOCK=\{tick\}$

And:

$CLOCK = (tick \rightarrow CLOCK)$


### 1.3 Guarded

**Suppose We have a recursive exquation**:

$X=X$

A process description which begins with a prefix is saided to be $guarded$, if $F(X)$ is a **guarded expression** containing the process name $X$, and $A$ is the alphabet of $X$, then we claim that the equation[3]:

$X=F(X)$


Has a **unique** solution with alphabet $A$, It's sometimes convenient to denote the **solution** by expression:

$\mu X: A \bullet F(X)$

**Examples**:

1)

$\alpha CH5A = \{in5p, out2p, out1p\}$

$CH5A=(in5p \rightarrow out2p\rightarrow out1p \rightarrow out2p\rightarrow CH5A)$

2)

$CLOCK = \mu X:\{tick\}\bullet\{tick \rightarrow X\}$

3) 

$\alpha VMS = \{coin, chos\}$

$VMS =(coin \rightarrow (choc \rightarrow VMS))$

Can be denote as

$VMS=\mu X: \{coin, choc\} \bullet (coin\rightarrow(choc \rightarrow X))$


### 1.4 Choice

if $x$ and $y$ are distinct events


$(x\rightarrow P | Y \rightarrow Q)$

And:

$\alpha(x\rightarrow P\ |\ y\rightarrow Q)=\alpha P \ | \{x,y\}\subseteq \alpha P$

The bar "|" should be pronoced "choice": "$x$ then $P$ choice $y$ then $Q$"

**Example**

1) A machine that servies either chocolate or tcoffee*


$VMCT=\mu X \bullet coin \rightarrow (choc \rightarrow X | toffee \rightarrow X)$

**If $B$ is any set of events and $P(x)$ is an expression defining a process of each different $x$ in $B$, then it can be denote as**:

$(x:B \rightarrow P(x))$

**Examples**

1） A process which at all times can engage in any event of its alphabet $A$

$\alpha RUN_A = A$

$\alpha RUN_A = (x:A \rightarrow RUN_A)$


2) In the special case taht menu contains only one event $e$:

$(x:{e} \rightarrow P(x)) = (e\rightarrow P(e))$

3) More special case taht inital menu is empty, and do nothing:

$(x:\{\}\rightarrow P(x))=(y:\{\}\rightarrow Q(y))=STOP$

**The binary choice operator | can be alwo be defined using the more general notation**

$(a\rightarrow P | b\rightarrow Q)=(x:B\rightarrow R(x))$

Where $B={a,b}$, and $R(x)=if\ x = a\ then\ P\ else\ Q$

### 1.5 Mutual Recursion

We can using index variables to specify mutal recursion

**Examples**

$CT_0=(up\rightarrow CT_1 | arrount \rightarrow CT_0)$

Which can be represent as:

$CT_{n+1} (up\rightarrow CT_{n+2} | arrount \rightarrow CT_{n})\ |\ n\in \mathbb{R}$

### 1.6 Laws

**L1**: $(x:A\rightarrow P(x))=(y:B\rightarrow Q(y)) \equiv (A=B\ \land\ \forall x: A\bullet P(x)=Q(x))$

**L2**: if $F(X)$ is a gugarded expression, $(Y=F(Y))\equiv (Y=\mu X \bullet F(X))$

**L3**: if $(\forall i: S\bullet (X_i=F(i,X)\ \lor \ Y_i=F(i, Y)))$ then $X=Y$, where:

$S$ is an indexing set with one member for each equation;

$X$ is an $array$ of processes with indices ranging over the set $S$ and $F(i, X)$ is a guard expression.

## II Implementation of processes[4]

A process can be written in the form:

$(x:B\rightarrow F(x))$

Where $F$ is a function from symbols to processes, and in the case of *recursion*, it can be written with guarded exp

\begin{gather}
\mu X \bullet (x:B \rightarrow F(x, X))
\end{gather}

and this may be unfolded to the requied from using L2

\begin{gather}
(x:B \rightarrow F(x, \mu B \bullet (x:B \rightarrow F(x, X))))
\end{gather}

Thus every process may be regarded as $function\ F$ with a domain $B$. For each $x$ in $B$, $F(.)$ defined the future behavior of the process if the first event was $x$.



We can present the process with Haskell Algebraic data types as:

```haskell
type Event = String
type Alphabet = [Event]

data Process a = Cons Event (Process a) | Stop
infixr 5 →
(→) = Cons

data Guard a = Guard Alphabet (Process a)
infixr 5 •
(•) = Guard

-- Event and Processes

coin = "coin"::Event
choc = "choc"::Event

prefix = coin → (choc → Stop)

-- Recursion

tick = "tick"::Event
clock = tick → clock

-- Guard
a = [coin, choc]::Alphabet

uX = a • prefix
```

And we can implement choice via type class:

```Haskell
class Choice a where
  (⏐) :: a -> a -> (Event -> a)

instance Choice (Process a) where
  a ⏐ b = \x -> if x == event(a) then a else b

```

## III Trace

A $trace$ of behavior of a process is a finite sequence of symbols recording the events in witch the process has engaged up to some moment in time[5], which will be denoted as sequence of symbols.

** Trace**: 

$\left<x, y\right>$ consist of two events, $x$, folled by $y$.

$\left<x\right>$ is a sequence containing only the event $x$.

$\left<\right>$ is the empty sequence containing no events.

** Operations**:

$s, t, u$, stand for traces

$S, T, U$ stand for sets of traces.

$f, g, h$ stand for functions

### 3.1 Catenation / Chain

The most important operation on traces is catenation, with constructs a trace from a apir of operands $s$ and $t$ by simply written:

$s^{\frown} t$

**Laws**

* **L1**: $s^{\frown}\left<\right>=\left<\right>^{\frown}s=s$

* **L2**: $s^{\frown}(t^{\frown}u)=(s^{\frown}t)^{\frown}u$

* **L3** $s^{\frown}t=s^{\frown}u\equiv t=u$

* **L4** $s^{\frown}t=u^{\frown}t\equiv s=u$

* **L5** $s^{\frown}t=\left<\right>\equiv s=\left<\right> \lor t= \left<\right>$

Let $f$ stand for a function with maps traces to traces. The function is said to be $strict$ if it maps the empty trace to empty trace:

$f(\left<\right>)=\left<\right>$

It is staid to be $distributive$ if it distribute through catenation

$f(s^{\frown}t)=f(s)^{\frown}f(t)$

All distributive function are strict.

if $n$ is natural number, we define $t^n$ as $n$ copies of $t$ catenate with each other. It's readily defined by intruceion on $n$

* **L6**: $t^0=\left<\right>$

* **L7, L8**: $t^{n+1}=t^{n\frown}t=t^{\frown}t^n$

* **L9**: $s^{\frown}t^{n+1}=s^{\frown}(t^{\frown}s)^{n\frown}t$

### 3.2 Restriction

The expression $(t \restriction A)$ denotes the trace $t$ when restricted to symbol in the set $A$; it is formed from $t$ simply by omitting all symbols outside A: Eg:

$\left<a, b, c, a\right> \restriction \{a, b\} = \left< a, b \right>$

### 3.3 Head and Tail

If $s$ is nonempty sequence, it's first sequence is denoted $s_0$, and result of removing the first symbol is $s'$. E.g:

$\left<x, y,x\right>_0=x$


$\left<x, y,x\right>'=\left<y,x\right>$

### 3.4 Star

The set $A^*$ is the set of all finite traces (including $\left<\right>$) which are formed from symbols in the set $A$. When such traces are restricted to $A$, thay remin unchanges:

$A^*={s\ |\ \restriction A=s}$

### 3.5 Ordering

If $s$ is a copy of an inital subsequence of $t$, it's possible to find some extension $u$ of $s$ such that $s^{\frown}u=t$. Thus we defined the oredering relation:

$s \leq t = (\exists u \bullet s^{\frown}u=t)$

### 3.6 Lengh

The length of the trace $t$ is denoted $\#t$. For examle:

$\#\left<x,y,z\right>=3$

### 3.7 Implementation

In Haskell, we can simply implement $trace$ via structure $List$.

# IV Reference

[1][2][3] Communicating Sequential Processes, C.A.R. Hoare, May 18 2015, chapter, 1.1

[4] Communicating Sequential Processes, C.A.R. Hoare, May 18 2015, chapter, 1.4

[5] Communicating Sequential Processes, C.A.R. Hoare, May 18 2015, chapter, 1.4

