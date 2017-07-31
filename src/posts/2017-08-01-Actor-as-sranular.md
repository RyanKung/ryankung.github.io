<h1><center>Actor as Sranular, A Isomorphism Micro-services Architecture</center></h1>


**<center>Ryan J.K</center>**
**<center>ryankung(at)ieee.org</center>**


## I Preface

**TL;DR：The problem is a distributed problem, thus Microservices Architecture is Bullshit**

-----



Micro-Services was introducted by Peter Rodgers and Juval Löwy in 2005 [1,2,3]. The philosophy of it essentially equals to the Unix philosophy of "Do one thing and do it well". [4][5][6]:


* The services are small - fine-grained to perform a single function.

* The organization culture should embrace automation of testing and deployment. This eases the burden on management and operations and allows for different development teams to work on independently deployable units of code.[7]

* The culture and design principles should embrace failure and faults, similar to anti-fragile systems.

* Each service is elastic, resilient, composable, minimal, and complete.[6]

And as Leslie Lamport's defination of distributed which in given in 2000, the micro-services architecture should always be a distributed system. In the viewpoint, a network of interconnected computers is a distributed system, and a single computer can be also be viewed as a distributed system in which the central control unit, the menory unit, and the input-output channels are sparate process. *A system is distributed is the message transmision delay is not negligible compared to the time between events in a single process*.[9]

In the fact, Addressing to the granularity of services, the realworld usecase of microservices architecture is usually either a distributed system or based on a distributed system such as some raft or paxos implementation like etcd[9], consul[10] and zookeeper[11].

Thus the "Unix Philosophy" of "Do one thing and do it well" is actually talking about the Philosophy of about the Distributed System archiecture", which is also descriping how "micro-services architecture" works.

## II None-causal modeling


**TL;DR: CPS is a none-causal language for processing modeling, and, Two Turing Award laureates thinks CSP (Actor model) is good for distributed system, and it's really so fucking good**

### 2.1 Think in the Domain
**TR;TD: DSLs is a model complete method for solving problem**


The most popular modeling method of micro-services nowadays is DDD (Domain-Driven Design), which trying to bind the model with the concrete implementation. The premise of the DDD is to make the modeling of the function or service be focus on the core domain and domain logic.[15][16]

When talking about Domain-Driven Design, we usually connected it with the DSL (Domain-Specific Languages). There are two ways in which the modeling can be understood:  descriptive and preciptive. A descriptive model represent an existing system Thus a presciptive model is one that can be used to construct the target system.[17]. DSLs always used prescriptive model as the term model, and the DDD is actually a equivalently descriptive modeling method.

In model theory, a first-order theory is called model complete if every embedding of models is an elementary embedding. Equivalently, every first-order formula is equivalent to a universal formula[18]. We known that if a DSLs is Turing Completed, then, we can call it as GPL(General Purpose Language). The GPL is acturally a model companion of The Turing Machine. 

And when we said that a language $l$ covers a subset of $P$, we can simply call this subset the $domain$ covered with $l$. The subset of $P$ in that domain $P_D$ is equal to the subset of $P$ we can express with a language $l$ $P_l$. So, we cannot ask a question like: "Does the language adequately cover the domain?", **since it always does, by definition**.[17] And the definition can be also interept as that "**DSL is always Model Complete, by definition**"[17]

### 2.2 None-Causal Modeling
**DL;DR: The Domain is leads to the question of how it works, but not how it looks like**

There is two kind of languages for modeling a complex system: **Causal (or block-oriented) languages** and **none-causal (or object-oriented) language**.[12] The drawback of Causal Languages is: Needing to explicty specify the causality, which hampers modularity and reuse[19]. None-causal language is tried to solve the issue of cause language via allowing the user to avoid committing the model it self to a specific causality.[20]

### 2.3 CSP and FHM

CSP (Communication Sequential Processes) is a typical None-Causal Language, for modeling the `Processes` of `Distributed System`, created by C.A.R Hoare, and still keeping update in nowadays(2015)[14]. It defined a process as this:

Let $x$ be an event and let $P$ be a process, Then $(x \rightarrow P)$ (proounced “$x$ then $p$”)

In 1983, when Lamport talk about CPS, he said: "It's a fine language, or more precisely, a fine set of communication constructs. Hoare deserved his **Turing award**...,We really know that CSP is the right way of doing things... "[14]. But He also thinks that "While theorieticians are busy studying CPS, people out there in the real world are building Ethernets. And CSP doesn't seem to me to be a very good model of Ethernets... CSP isn't a very good language for describing this kind of algorithm(The MUTEX Problem), although it's good for other kinds of algoithms."[14]

Now we knew that the problem of distributed system in 1983 is the consensus problem. Which is means how do processes learn that a shared value was used or selected. In fact that in 1978 the core solution had already introduced by Leslie Lamport: The algorithms based on the logic timestamp(**Lamport Timestamp**)[21], but the algorithm hasn't be applied until 1990, the year of invention of Paxos [22]. (which algorithm is worthing a Turing Award).

There is some other researchs based on **TimeStamp**, In research of Yale, they has developed a framework called *functional rective programming*, or FRP[8]. which is highly suited fro causal hybird modeling[9]. And, because the full power of a functional language is avaliable, it exhibits a high degree of modularity, allowing reuse of components and design patterns.[23] And *functional hybird modeling*, or FHM is a combined of FRP and none-causal languages. Which can be seen as a generalization of FRP, since FRP's functions on singals are a special case of FHM's relations on signals. FHM, like FRP, also allows the description of structurally dynamic models.[24]

And the same two key ideas between CSP and FRP are to give first-class status to relations on signals/messages and to provide constructs for discrete switch ing between relations.

## III Isomorphism with Actor Model

### 3.1 Isomorphism Graphic

Lets recall that how people talk about the micro-services when they are talking about micro-services: Small, Testable, Robustness, Composable and Elastic. The graphic of the micro-services system should be like a vertex-edge map, all vertexs are symmetic thus they send messages to each orther for implement a complete function(figure 3.1). The vertex $S_i$ denotes the services, and the length $l$ of edgo of $(S_i, S_j)$ denotes the cost time between services $S_i,S_j$. With CSP modeling, the processing between $S_i,S_j)$ can be also present as $(S_j \rightarrow S_j)$.

<img src="https://ryankung.github.io/images/fig_3_1.png" /><center>figure 3.1[25]</center>


And some Services $S_i$ maybe multi-processes and should have it's workers like this.

<img src="https://ryankung.github.io/images/fig_3_2.png" /><center>figure 3.2[25]</center>

In Figure 3.2., Services $S_2$ have three workers $w_i; i\in[1,3]$. We can see that all vertexs are Isomorphism. 

### 3.2 Actor Model

Actor Model is an implementation of CSP (or lamport-timestamp based distributed FRP), invented by Carl Hewitt [24], which is also one of embers of 1970s AI wave (The second wave of AI). In Actor model, the model of processes of CSP are defined and structed with `Actor`, the actors can make local decisions, create more actors and send/response messages. The most famous implementation of Actor Model language is Erlang. Some people think golang's goroutine or Python3's coroutine are also implementation of CSP or Actor model, but actually they not, because of the mutable statement and memory sharing.

In actor model, the `Actors` whose controlling the event loop is called `Arbiter`. Thus the `Actors` whose sharing the IO loop of `Arbiter` and controll other `Actors as Workers` is called `Monitor`. So a classic `Arbiter-Monitor`-`Actor` may like this:

<img src="https://ryankung.github.io/images/monitors.svg" /><center>figure 3.3[25]</center>

Thus the figure 3.2 should be represent as:

<img src="https://ryankung.github.io/images/fig_3_4.png" /><center>figure 3.4[25]</center>

The Arbiter Actor here is actually a implementation of the vertex-edge map we descripted in Ch. 3.1. In real-world of enginering, the Arbiter maybe implement by Operator System itself, the VM, or just based on the network.


### 3.3 Lamport timestamp based FRP

If we think that a distributed FRP is just the FRP system based on lamport timestamp but not the real-word timestamp, then we can found that there is alot of FRP features in the current Actor Model based Micro-services system, Such as Composable and Elastic. 

We can remodel the CSP actor model $(S_i \rightarrow S_j)$ with $( \rightarrow_{(t_n)} S_i(t,m;\theta) \circ S_j(t,m;\theta))$. Where $\rightarrow$ denotes the function of event stream. and the $S_i(t,m;\theta)$ is means that a monitor actor is listen to the message stream $(t,m)$, and it's elastic rate is based on param $\theta$. And with builted in lamport timestamp, we won't meet the problem of consensus, because we will actuall knowns that which vectex is outdated or not.


## Reference


[1] Rodgers, Peter. "Service-Oriented Development on NetKernel- Patterns, Processes & Products to Reduce System Complexity Web Services Edge 2005 East: CS-3". CloudComputingExpo 2005. SYS-CON TV. Retrieved 3 July 2017.

[2] Löwy, Juval (October 2007). "Every Class a WCF Service". Channel9, ARCast.TV.

[3] Löwy, Juval (2007). Programming WCF Services 1st Edition. pp. 543–553.

[4] Lucas Krause. Microservices: Patterns and Applications. ASIN B00VJ3NP4A.

[5]Lucas Krause. "Philosophy of Microservices?".

[6]Jim Bugwadia. "Microservices: Five Architectural Constraints".

[7] Li, Richard. "Microservices Essentials for Executives: The Key to High Velocity Software 
Development". Datawire. Datawire, Inc. Retrieved 21 October 2016.


[8] Leslie Lamport, Times ,Clocks, and the Ordering of Events in a Distributed System


[9] A distributed, reliable key-value store for the most critical data of a distributed system. https://coreos.com/etcd/

[10] Service Discovery and Configuration Made Easy https://www.consul.io/

[11] Apache ZooKeeper is an effort to develop and maintain an open-source server which enables highly reliable distributed coordination. https://zookeeper.apache.org/

[12] [Andrew Kennedy. Programming Languages and Dimensions. PhdD thesis, University of Cambridge, Computer Laboratory, April 1996. Published as Technical Port No. 391.](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-391.pdf)

[13] C.A.R Hoare, Communicatiing Sequential Process, May 18, 2015.

[14] Leslie Lamport, 1983 Invited Address, Solved Problems, Unsolved Problems and Non-problems in Concurrency.

[15] Domain-driven design
 http://dddcommunity.org/
 
[16] Evans, Eric (2004). Domain-Driven Design: Tackling Complexity in the Heart of Software. Addison-Wesley. ISBN 978-032-112521-7. Retrieved August 12, 2012..

[17] DSL Engineering, Designing, Implementing and Using Domain-Specific Languages, Markus Voelter, dslbook.org

[18] Chang, Chen Chung; Keisler, H. Jerome (1990) [1973], Model Theory, Studies in Logic and the Foundations of Mathematics (3rd ed.), Elsevier, ISBN 978-0-444-88054-3

[19] Frannois E. Cellier. Object-oriented modelling: Means of dealing with system complexity. In Proceeedings of the 15th Benelur Meeting on Systems and Control, Mierlo, The Netherland, pages 53-64, 1006 cited in Functional Hybird Modeling, Henrik Nilsson, John Peterson, and Paul Hudak, Department of Computer Science, Yale University, PADL 2003

[20] Henrik Nisson, John Perterson, and Paul Hudak, Department of Computer Science, Yale University. Functional Hybird Modeling

[21] Leslie Lamport, Time, Clocks and The Ordering of Events in a Distributed System. Jul 1978

[22] Leslie Lamport, The Simple Paxos, 2000

[23] Zhanyong Wan and Paul Hudak. Functional reactive programming from first princple. In proceeding s of PLDI'01: Symposium on Programming Language Design and Implementation, pages 202-202, June 2000.

[24] Henrik Nilsson, John Peterson, and Paul Hudak, Functional Hybird Modeling, Department of Computer Science, Yale University, PADL 2003

[24] Carl Hewitt; Peter Bishop; Richard Steiger (1973). "A Universal Modular Actor Formalism for Artificial Intelligence". IJCAI.

[25] Design of Pulsar, A Actor Model based framework, http://quantmind.github.io/pulsar/design.html

