# <center>Actor Model :: Actix</center>

<center>Ryan Kung</center>
<center>ryankung#ieee.org</center>



Actix is a Actor Model implementation which was first presented by [Carl Hewitt, ...] at 1973 as `A Universal Modular ACTOR Formalism for Artificial Intelligence` for `PLANNER` Project. Actor model can be defined in terms of one kind of behavior: 

"sending messages to actors. An actor is always invoked uniformly in exactly the same way regardless of whether it behaves as a recursive function, data structure, or process."[1]

Which is means that an Actor is a computational entity that, in response to a message it receives, can concurrently[4]:

* Send a finite number of messages to other Actors;
* Create a finite number of new Actors;
* Designate the behavior to be used for the next 
message it receives.

For Send and Handle message, Actors have a abstract structure named Mailbox, which is the place where received messages are stored. Although multiple actors can run at the same time, an actor will process a given message sequentially, this means that if you send 3 messages to the same actor, it will just execute one at a time. To have these 3 messages being executed concurrently, you need to create 3 actors and send one message to each. Those messages are stored in other actors' mailboxes until they're processed.[5]

## Actix

`Actix-rs` is an actor model implementaion in Rust programming language.

### actix::Actor

Actors are objects which encapsulate state and behavior. 

Actors run within specific execution context `Context`. Context object is available only during execution. Each actor has separate execution context. Also execution context controls lifecycle of an actor:
(`Started`, `Running`, `Stoping`, `Stopped`).

#### Started

Actor starts in Started state, during this state started method get called.

#### Running

After Actor's method started get called, actor transitions to Running state. Actor can stay in running state indefinitely long.

#### Stopping:

Actor execution state changes to stopping state in following situations:

1. Context::stop get called by actor itself.
2. All addresses to the actor get dropped
3. no evented objects are registered in context.

Actor could restore from `stopping` state to running state by creating new address or adding evented object, like future or stream, in `Actor::stopping` method.

#### Stopped

If actor does not modify execution context during stopping state actor state changes to Stopped. This state is considered final and at this point actor get dropped.

### actix::arbiter

Arbiter is an event loop controller, Arbiter controls event loop in its thread. Each arbiter runs in separate thread. Arbiter provides several api for event loop access. Each arbiter can belongs to specific `System` actor. By default, a panic in an Arbiter does not stop the rest of the System, unless the panic is in the System actor. 

### actix::sync::AyncArbiter

Sync actors could be used for cpu bound load. Only one sync actor runs within arbiter's thread. 

Sync actor process one message at a time. 

Sync arbiter can start multiple threads with separate instance of actor in each. Note on actor stopping lifecycle event, sync actor can not prevent stopping by returning false from stopping method. 

Multi consumer queue is used as a communication channel queue. To be able to start sync actor via SyncArbiter Actor has to use SyncContext as an execution context.

### actix::Supervised

Actors with ability to restart after failure, Supervised actors can be managed by Supervisor. Lifecycle events are extended with restarting method. If actor fails, supervisor creates new execution context and restarts actor. restarting method is called during restart. 

### Ref:

> 1. Carl Hewitt, ..., A Universal Modular ACTOR Formalism for Artificial Intelligence.

> 2. J.C.M. Baeten, A brief history of process algebra

> 3. Colin Fidg, Software Veriﬁcation Research Centre Department of Computer Science The University of Queensland Queensland 4072, Australia, A Comparative Introduction to CSP, CCS and LOTOS, 1994

> 4. Carl Hewitt, Actor Model of Computation

> 5. The Actor Model, https://www.brianstorti.com/the-actor-model/
> 6. Actix API Document https://actix.rs/actix/actix/trait.Actor.html