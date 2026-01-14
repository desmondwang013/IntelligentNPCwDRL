
---

# Project Design Document

**Real-Time Instruction-Following NPC with Deep Reinforcement Learning (MVP)**

## Project Vision and Motivation

This project aims to build a real-time, playable NPC in a 2D environment that can follow natural language instructions in a way that feels humane, generalizable, and trustworthy. The NPC should be responsive to user intent, capable of handling interruptions, and able to operate without relying on pre-scripted goals or rigid command parsing.

Rather than assuming the user’s intent can be fully decoded into predefined slots, the system is designed to learn instruction grounding end-to-end through Deep Reinforcement Learning (DRL). The NPC may hesitate, ask for clarification, or make imperfect but understandable decisions, as a human assistant might. The focus is not on optimality, but on believable competence and adaptability.

The system is intentionally designed to be engine-agnostic. A Python “brain” is developed first, defining all intelligence, learning, and decision-making. A game engine is integrated later as a thin rendering and input layer, without changing the core logic.

---

## High-Level Architecture

The system consists of two conceptual layers.

The first is the Python Brain, which is authoritative during MVP development. It contains the world simulation, intent management, DRL policy, deterministic reward computation, logging, training, and evaluation hooks.

The second is the Game Engine layer, introduced later. Its responsibility is rendering, user interface, and visual feedback. It forwards user input events to the Python layer and displays the resulting world state. As long as it respects the observation and action contracts defined by the Python brain, any engine can be used without retraining the model.

---

## Runtime Model

The system operates in a continuous, real-time session rather than episodic interactions. The world advances at a fixed rate of 16 ticks per second. The NPC continuously selects actions at each tick. The user may issue new instructions at any time, including while the NPC is executing a previous instruction.

There is no concept of “one instruction per episode.” Instead, instructions arrive as events, and the NPC must adapt immediately. This allows natural interactions such as giving a command, interrupting it, or changing one’s mind mid-execution.

---

## Intent System

Each user instruction is represented internally as an intent. Only one intent may be active at any time. When a new instruction arrives, it preempts the current intent immediately.

An intent progresses through one of several terminal states. It may be completed successfully, canceled explicitly by the user, or terminated by the system due to timeout, lack of progress, or impossible constraints. Canceled intents are not treated as failures.

Intent completion is defined deterministically using world state. For position-based intents, completion requires the relevant distance threshold to be met and remain stable for three consecutive ticks. Time limits are adaptive rather than fixed, scaled by task difficulty and recent progress. If the NPC is near success, a small grace window allows extra time to avoid frustrating near-miss failures.

---

## World Specification

The MVP world is a deterministic 2D grid of size 64×64. All movement and interaction occurs in abstract world units rather than pixels.

The world contains a user entity, an NPC entity, and ten objects. Objects are spawned randomly at reset with a minimum separation distance to avoid overlaps or ambiguity.

Each object has a stable object ID that is always rendered visibly. In addition, each object has basic appearance attributes: a type (fixed, light, or heavy), a color, a shape, and a size (small, medium, or large). These attributes are deliberately simple but sufficient to support natural references such as “the blue triangle” and to enable visual evaluation by humans and vision-language models.

---

## Action Space

The NPC has a small discrete action space designed for learnability and clarity. The actions are move up, move down, move left, move right, wait, and speak.

Movement advances the NPC by 0.5 world units per tick, subject to collision rules. The wait action holds position for four ticks and is the primary mechanism for expressing slow or cautious behavior. The speak action consumes the current tick and produces a short verbal response, such as an acknowledgment or clarification request.

---

## Observation Design

The policy receives a fixed-length numerical observation vector at every tick. This includes the NPC position, the user position, and their relative offset. It also includes information about the nearest eight objects, chosen by proximity. For each of these objects, the observation includes relative positions to both the NPC and the user, along with categorical identifiers for type, color, shape, and size, as well as movable and collider flags.

The observation also includes intent context. The active intent’s raw text is embedded using a local transformer-based embedding model and cached. The intent age in ticks is included so the policy can reason about urgency or prolonged execution. An optional focus pointer provides a soft hint toward a likely referenced object when applicable, without enforcing any rigid decoding.

---

## Language Handling and Clarification

Instruction interpretation is learned end-to-end. There is no rule-based planner or slot-filling parser in the control loop. Text is converted to numerical embeddings, and the policy learns to condition its behavior on these embeddings and the world state.

Clarification is treated as a runtime behavior rather than a preprocessing step. The NPC may choose to speak when references are ambiguous, when constraints conflict with the world, or when progress stalls. When a clarification is issued, the current intent pauses and waits for user input, which either resolves the ambiguity or replaces the intent entirely.

---

## Reward and Evaluation System

The system uses two complementary evaluation layers that operate on different timescales and serve different purposes.

The first layer is deterministic evaluation, also referred to as the reward function. This layer runs continuously during gameplay and training, at every tick. It is numeric, fast, and fully automatic. It provides the core learning signal for DRL and never blocks runtime.

Deterministic reward aspects include progress toward the active intent, mild time pressure, collision penalties, anti-stuck and anti-oscillation penalties, override compliance, and a small cost for speaking. Event-level rewards are applied when an intent ends, such as a completion bonus, penalties for timeouts or impossibility, and bonuses for justified clarification. These aspects are modular and layered; exact formulas and weights are intentionally left tunable at this stage.

The second layer is semantic evaluation. This layer runs offline after an intent finishes and is selective rather than continuous. Its purpose is to capture language-grounded correctness that numeric metrics cannot express, such as whether the NPC moved “slowly enough,” avoided a region, obeyed an override promptly, or asked for clarification appropriately.

Semantic evaluation operates at the intent level. After an intent ends, an evaluation bundle is generated. This bundle contains the intent text, intent type, start and end ticks, relevant object references with their visual attributes, numeric summaries such as distance change and wait ratio, and visual evidence in the form of a small set of frames captured at key moments.

Evaluation is performed by two independent judges. A local vision-language model evaluates a larger automated slice of intents, including all linguistically complex or constraint-heavy cases and a random audit sample. A human evaluator labels a smaller, stratified slice to serve as calibration and ground truth. The two slices partially overlap. Disagreements, low-confidence cases, or benchmark intents may be escalated to a rare external auditor.

Both machine and human evaluators produce structured outputs consisting of a success or failure decision, a normalized score, a confidence value, and categorical reason tags. These labels are stored by intent ID.

Semantic evaluation does not affect gameplay directly. Instead, during training updates, semantic scores are looked up by intent ID and injected as additional reward signals, typically as an intent-terminal bonus. If no semantic label exists for an intent, no semantic reward is applied. Deterministic reward always runs, regardless of whether semantic evaluation occurs.

This separation ensures that gameplay remains real-time and responsive, while learning benefits from richer supervision when available.

---

## Training Model

The NPC is trained using Proximal Policy Optimization. Because runtime is continuous, training does not rely on episodes. Instead, fixed-length segments of ticks are collected. Intents may begin or end inside these segments.

Training proceeds in cycles: rollouts are collected, segments and intents are logged, a subset of intents is evaluated semantically, labels are merged, and the policy is updated. Evaluation and training are decoupled in time.

---

## Engine Compatibility Strategy

During MVP development, the Python layer is authoritative. The world simulation, reward computation, and learning all occur in Python. The engine, when introduced, acts as a renderer and input forwarder.

A small, stable contract is defined between the engine and the Python brain: a fixed observation schema, a discrete action ID space, and an event stream for user input and intent boundaries. As long as this contract is respected, engine integration will not require redesigning or retraining the core model.

---

## MVP Development Plan

The agreed development sequence is as follows. First, implement the Python world simulator with spawning rules and object attributes. Next, implement the intent manager and event stream. Then build the observation vector and action schema. After that, validate the runtime loop using a random or stub policy. Logging and segment slicing follow. PPO training is then integrated, and evaluation hooks are added without tuning. Engine integration and reward tuning are deferred until the MVP loop is fully operational.

---

## Current Status

At this stage, the project architecture, runtime model, intent system, observation design, action space, reward aspects, and evaluation pipeline are fully specified at the conceptual level. Reward formulas and weights remain intentionally flexible. The project is ready to proceed into MVP implementation without risking incompatibility with future engine integration.

---

If you want, the next step can be to turn this into a versioned `design.md` for the repository, or to scaffold code modules that mirror this document one-to-one.
