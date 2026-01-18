# Intelligent NPC with Deep Reinforcement Learning

A Python-based system for training an NPC that follows natural language instructions in real-time.

---

## Project Overview

This project builds the "brain" of an NPC that can understand and execute natural language commands like "go to the blue triangle", "pick up the red box", or "bring that to me". The architecture separates **language understanding** from **action execution**:

- **LLM** handles language interpretation, ambiguity resolution, and dialogue
- **RL agent** handles physical execution of structured commands (navigation, interaction)

The RL agent does not parse language directly — it receives structured commands (e.g., `NAVIGATE target_id=obj_3`) and learns robust execution skills. This separation allows each component to do what it does best: LLMs excel at semantic understanding, while RL excels at continuous control and handling physical uncertainty.

The system is designed to be **engine-agnostic** — the Python layer handles all the thinking, and a game engine (like Unity) can be plugged in later just for rendering.

---

## High-Level Architecture (End Goal)

```
┌─────────────────────────────────────────────────────────────────┐
│  USER                                                           │
│  "Go to the red box and pick it up"                             │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLM Layer                                                      │
│  - Interprets natural language                                  │
│  - Handles ambiguity ("which red box?")                         │
│  - Asks clarifying questions if needed                          │
│  - Decomposes complex instructions into steps                   │
│                                                                 │
│  Output: Structured ActionPlan JSON                             │
│  {                                                              │
│    "actions": [                                                 │
│      {"type": "NAVIGATE", "target": "red box"},                 │
│      {"type": "PICK_UP", "target": "red box"}                   │
│    ]                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Validator + Resolver (Deterministic)                           │
│  - Validates action types are supported                         │
│  - Resolves "red box" → obj_3 (exact match on world state)      │
│  - Checks target exists and is unambiguous                      │
│  - Returns error/clarification request if invalid               │
│                                                                 │
│  Output: Resolved ActionPlan                                    │
│  {                                                              │
│    "actions": [                                                 │
│      {"type": "NAVIGATE", "target_id": "obj_3"},                │
│      {"type": "PICK_UP", "target_id": "obj_3"}                  │
│    ]                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  RL Executor (Motor Skills)                                     │
│  - Receives: action_type + target_id                            │
│  - Observation: positions, obstacles, goal (NO language)        │
│  - Executes physical actions (navigate, pick up, etc.)          │
│  - Handles dynamics, collision avoidance, uncertainty           │
│  - Reports success/failure                                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  World State                                                    │
│  - Updated positions, inventory, etc.                           │
│  - Outcome fed back to LLM for narration or replanning          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **No language embeddings in RL observation** — The RL agent receives only structured goal information (target position, action type). Language understanding is fully delegated to the LLM.

2. **Deterministic target resolution** — "red box" → `obj_3` is a lookup/match operation, not learned. This ensures predictable behavior and clear error handling.

3. **Optional: Semantic parser** — If needed, a lightweight semantic layer can sit between LLM output and the validator to handle format variations. But the LLM should be prompted to output valid JSON directly.

### Why RL Is Not Used for Language Parsing

| Concern | LLM Approach | RL Approach |
|---------|--------------|-------------|
| Sample efficiency | Pre-trained on billions of tokens | Needs millions of examples per concept |
| Ambiguity handling | Can ask clarifying questions | Silent failure or random guessing |
| Generalization | Handles novel phrasings naturally | Requires retraining for new language |
| Dialogue | Native capability | Not designed for multi-turn interaction |

RL is powerful for **continuous control** — navigating around obstacles, timing interactions, handling physics. Language understanding is better left to models designed for it.

---

## Vision vs. Current State

### The Goal

An NPC that truly understands language and acts on it:

```
User: "Go to the red box"       → NPC navigates
User: "Pick up the wheat"       → NPC navigates + picks up
User: "Attack the enemy"        → NPC approaches + fights
User: "Bring the box to me"     → NPC navigates + picks up + returns + drops
```

This is **instruction-following** — the NPC must learn:
1. What action type is requested (go / pick up / attack / haul)
2. What object(s) are involved
3. How to sequence low-level actions

### Current State (MVP)

The current implementation tests the **navigation scaffold** only:

```
Training demo:
1. System picks a random object (e.g., obj_3 = red box)
2. System generates instruction: "Go to the red box"
3. System provides both text AND target_id to the agent
4. Agent learns to navigate to the given target
```

**What's missing:** A parser that figures out "red box" → `obj_3` from user text alone. The training currently bypasses this by providing the answer.

**What current MVP proves:**
- World simulation works
- Embedding pipeline works
- RL training loop works
- Navigation behavior can be learned

**What it doesn't prove yet:**
- Language understanding (mapping text → correct object)
- Action type inference (go vs pick up vs attack)

### Strategic Direction

We're building for a **layered architecture** where each component handles what it does best:

| Layer | Responsibility | Technology |
|-------|----------------|------------|
| Language Understanding | Interpret user intent, handle ambiguity, dialogue | LLM (future) |
| Target Resolution | Map descriptions → concrete IDs, validate | Deterministic/Supervised |
| **Execution (Current Focus)** | Navigate, interact, handle physics | RL (PPO) |

This is more scalable and robust than trying to make RL learn language semantics:
- LLMs handle "grab" ≈ "pick up" naturally through pre-training
- RL focuses purely on motor control and robust execution
- Target resolution is a well-defined supervised problem

**Phased Roadmap:**

```
Phase 1 (Current):
├── Scope: RL navigation execution only
├── Input: Structured command with target_id
├── Reward: Distance-based progress + completion
└── Goal: Prove RL can learn robust navigation

Phase 2:
├── Add target resolution layer
├── Extend action space (pick up, drop, etc.)
├── RL receives structured commands for each action type
└── Still automated reward

Phase 3:
├── Integrate LLM for language understanding
├── LLM → Target Resolver → RL pipeline complete
├── Complex instructions ("bring X to Y") decomposed by LLM
└── Human feedback for edge cases
```

---

## Project Structure

```
src/
├── world/              # The game world simulation
├── intent/             # Instruction management and completion tracking
├── observation/        # What the NPC "sees" each moment
│   ├── builder.py      # Legacy 575-dim observation (with embeddings)
│   └── simple_builder.py  # Simplified 24-dim observation (recommended)
├── runtime/            # Tying everything together
├── reward/             # Learning signals (with phased configs)
└── training/           # PPO training pipeline
    ├── environment.py  # Legacy NPCEnv (575-dim)
    ├── simple_env.py   # SimpleNPCEnv (24-dim, recommended)
    └── curriculum_v2.py # Curriculum learning controller

train_simple.py         # Main training script (recommended)
visualize_agent.py      # Watch agent behavior in real-time
```

---

## Part 1: World (`src/world/`)

**What it does:** Simulates the 2D game environment.

### Classes and How They Work

**`Position`** — A simple container holding x and y coordinates. It knows how to calculate distance to another position and can convert itself to a dictionary for serialization.

**`Entity`** — The base class for anything that exists in the world. Every entity has a position and a unique ID. Both NPC and User inherit from this.

**`NPC`** and **`User`** — Represent the two controllable characters. They're mostly identical for now (both have a speed of 0.5 units per tick), but they're separate classes because they'll diverge later — the NPC is controlled by the policy, while the User is controlled by external input.

**`WorldObject`** — Represents the interactive objects in the world. Each object has four attributes:
- **Type** (fixed/light/heavy) — determines if it can be moved
- **Color** (red/blue/green/yellow/purple/orange) — for visual identification
- **Shape** (circle/triangle/square/diamond) — for visual identification
- **Size** (small/medium/large) — affects collision radius

**`Spawner`** — Handles placing entities randomly at the start. It ensures nothing spawns too close together by checking distances against a minimum separation threshold. If it can't find a valid spot after 100 attempts, it gives up and raises an error.

**`World`** — The main class that ties everything together. It holds the NPC, User, and all objects. When you call `step(action)`, it:
1. Looks up the movement delta for that action (e.g., MOVE_RIGHT = +0.5 in x)
2. Calculates the new position
3. Checks if that position would collide with anything
4. If no collision, moves the NPC; if collision, stays in place
5. Increments the tick counter
6. Returns the new world state as a dictionary

The `get_state()` method returns everything as a nested dictionary — this is what gets sent to Unity or used by the observation builder.

### How They Connect

```
Spawner creates → User, NPC, [WorldObjects]
                         ↓
                  World holds all of them
                         ↓
            World.step() moves NPC, checks collisions
                         ↓
            World.get_state() → dictionary for external use
```

**The setup:**
- A 64×64 grid world
- One NPC (the agent we're training)
- One User (the human giving instructions)
- Ten objects scattered around (triangles, squares, circles of different colors and sizes)

**How movement works:**
- The NPC can move up, down, left, or right (0.5 units per tick)
- The NPC can also wait (do nothing) or speak
- Objects and entities can't overlap — there's collision detection

**What it provides:**
- `world.step(action)` — advance the world by one tick with the NPC's chosen action
- `world.get_state()` — get a snapshot of everything (positions, objects, etc.) as a dictionary
- `world.reset()` — start fresh with new random object placements

**Why it matters:** This is the environment the NPC lives in. The NPC takes actions here, and we measure success based on what happens here.

---

## Part 2: Intent (`src/intent/`)

**What it does:** Manages user instructions and tracks completion criteria.

> **Note: This module is transitional.** In the current MVP, intent handling includes text embeddings passed directly to the RL agent. In the intended architecture, language understanding will be handled by an LLM layer, and this module will simplify to a structured command interface. The RL agent will eventually receive only structured goals (target_id, action_type), not raw language embeddings.

### Classes and How They Work

**`TextEmbedder`** — Wraps a sentence-transformer model (all-MiniLM-L6-v2). When you give it a string like "go to the blue triangle", it returns a 384-dimensional numpy array that captures the semantic meaning. It caches results so embedding the same text twice doesn't hit the model again. The model loads lazily — it won't download or initialize until you actually embed something.

**`CompletionCriteria`** — Defines what "success" means for an intent. It holds:
- A target (either an entity ID like "obj_3" or a position like (10, 20))
- A distance threshold (how close is "close enough")
- A stability requirement (must stay close for N consecutive ticks)

When you call `check()`, it looks up the target's current position in the world state, calculates distance to the NPC, and returns True if within threshold.

**`Intent`** — Represents a single instruction. It holds:
- The original text
- The cached embedding (so we don't re-embed every tick)
- The status (active/completed/canceled/timeout)
- Timing information (when it started, when it ended)
- The completion criteria
- An optional "focus hint" pointing to which object is probably relevant

It also has an `get_adaptive_timeout()` method that extends the deadline if the NPC is making good progress (within 80% of the goal = 25% more time).

**`IntentManager`** — The controller that handles intent lifecycle. It:
- Creates new intents (and cancels any existing one first — only one active at a time)
- Calls the embedder to get text vectors
- Checks completion criteria every tick
- Tracks a stability counter (intent only completes after 3 consecutive ticks within threshold)
- Logs all intent events for later analysis

When `update()` is called each tick:
1. Check if we've exceeded the timeout → terminate as TIMEOUT
2. Check if completion criteria is met → increment stability counter
3. If stable for 3 ticks → terminate as COMPLETED
4. If moved away from target → reset stability counter

### How They Connect

```
User text → TextEmbedder → 384-dim embedding
                              ↓
IntentManager.new_intent() creates Intent with embedding
                              ↓
Every tick: IntentManager.update() checks CompletionCriteria
                              ↓
            Returns event if intent terminated
```

**The core idea:**
- When the user says something like "go to the red square", that becomes an **Intent**
- Only one intent is active at a time
- If the user gives a new instruction, the old one is immediately canceled (preemption)

**How instructions become numbers:**
- The text is converted into a 384-dimensional vector using a small language model (all-MiniLM-L6-v2)
- This vector captures the meaning of the instruction
- The NPC's policy network receives this vector as input

**Intent lifecycle:**
- **Active** — the NPC is working on it
- **Completed** — the NPC reached the goal (stayed near target for 3 ticks)
- **Canceled** — the user gave a new instruction or said "stop"
- **Timeout** — took too long (with some grace period if close to success)

**What it provides:**
- `manager.new_intent(text, ...)` — create a new instruction
- `manager.update(tick, world_state)` — check if the intent is complete or timed out
- `manager.get_current_embedding()` — get the 384-dim vector for the policy

**Why it matters:** This is how natural language enters the system. The NPC doesn't see words — it sees numerical embeddings that represent meaning.

---

## Part 3: Observation (`src/observation/`)

**What it does:** Packages everything the NPC needs to know into a single fixed-length array.

### Two Observation Builders

**`SimpleObservationBuilder` (Recommended — 24 dimensions)**

This is the streamlined observation for the RL executor architecture. It provides ONLY structured goal information:

| Component | Dimensions | Description |
|-----------|------------|-------------|
| NPC position | 2 | Normalized (x, y) |
| Target position | 2 | Normalized (x, y) |
| Direction to target | 2 | Unit vector pointing toward goal |
| Distance to target | 1 | Normalized distance |
| Target reached flag | 1 | 1.0 if within threshold |
| Nearest obstacles | 15 | 5 obstacles × 3 (relative x, y, radius) |
| Intent age | 1 | Normalized time since instruction |

**No language embeddings.** The RL agent receives structured goals, not text. Language understanding is delegated to the LLM layer.

**`ObservationBuilder` (Legacy — 575+ dimensions)**

> **Note:** This builder includes language embeddings (384 dimensions) and was used for earlier experimentation. It remains available but is not recommended for the current architecture.

### Classes and How They Work

**`ObservationConfig`** — Holds settings like world size, how many objects to include, embedding dimension, and the maximum intent age (for normalization). You can customize these, but the defaults match the world module.

**`ObservationBuilder`** — The main class that constructs observation vectors. When you call `build()`, it:

1. **Extracts positions** — pulls NPC and User positions from the world state dictionary
2. **Normalizes coordinates** — divides by world size so everything is in [0, 1] range
3. **Computes relative offset** — calculates (user - npc) so the NPC knows which direction the user is
4. **Sorts objects by distance** — finds the 8 closest objects to the NPC
5. **Encodes each object** — for each of the 8 nearest:
   - Relative position to NPC (2 floats)
   - Relative position to User (2 floats)
   - One-hot encoding of type (3 floats)
   - One-hot encoding of color (6 floats)
   - One-hot encoding of shape (4 floats)
   - One-hot encoding of size (3 floats)
   - Is movable flag (1 float)
   - Collision radius normalized (1 float)
6. **Appends intent embedding** — the 384-dim vector from IntentManager
7. **Appends intent age** — normalized by max age so it's in [0, 1]
8. **Appends focus hint** — one-hot over the 8 object slots indicating which one is the target

The result is a single numpy array of 575 float32 values, ready for a neural network.

**One-hot encoding** — categorical values like "red" or "triangle" are converted to binary vectors. For example, color has 6 options, so "red" becomes [1,0,0,0,0,0] and "blue" becomes [0,1,0,0,0,0]. This lets the neural network treat categories properly instead of assuming "blue > red" numerically.

### How It Connects

```
World.get_state() → dictionary of positions and objects
                              ↓
IntentManager.get_current_embedding() → 384 floats
IntentManager.get_intent_age() → 1 float
                              ↓
ObservationBuilder.build() combines everything
                              ↓
numpy array [575 floats] → ready for policy network
```

**What the NPC "sees" each tick (575 numbers total):**

1. **Its own position** (2 numbers) — where am I?
2. **The user's position** (2 numbers) — where is the human?
3. **Relative offset to user** (2 numbers) — how far is the human from me?
4. **The 8 nearest objects** (176 numbers) — for each object:
   - Where is it relative to me?
   - Where is it relative to the user?
   - What type is it? (fixed/light/heavy)
   - What color? (red/blue/green/yellow/purple/orange)
   - What shape? (circle/triangle/square/diamond)
   - What size? (small/medium/large)
   - Can it be moved?
5. **The current instruction embedding** (384 numbers) — what am I being asked to do?
6. **How long the instruction has been active** (1 number) — am I taking too long?
7. **Focus hint** (8 numbers) — which of the 8 objects is probably the target?

**What it provides:**
- `builder.build(world_state, embedding, age, focus)` — create the observation vector
- `builder.observation_dim` — tells you the total size (575)

**Why it matters:** Reinforcement learning policies need fixed-size numerical inputs. This module converts the messy real world into a clean vector the policy can process.

---

## Part 4: Runtime (`src/runtime/`)

**What it does:** Runs the main loop that ties everything together.

### Classes and How They Work

**`Event`** — A simple data container representing something that happened. It has a type (like NEW_INSTRUCTION or INTENT_COMPLETED), a tick number, and a data dictionary with details. Events are used both for user input and for logging what happened.

**`EventType`** — An enum listing all possible events:
- User inputs: NEW_INSTRUCTION, CANCEL_INSTRUCTION, USER_MOVE
- System events: INTENT_STARTED, INTENT_COMPLETED, INTENT_CANCELED, INTENT_TIMEOUT
- NPC events: NPC_SPEAK

**`EventQueue`** — A buffer for incoming events. When the user (or Unity) wants to give an instruction, it gets pushed to the queue. At the start of each tick, the runtime pops all pending events and processes them. This decouples input timing from the tick loop — you can queue multiple events between ticks if needed.

**`RuntimeConfig`** — Holds settings like ticks per second, random seed, default intent timeout, and reward configuration.

**`StepResult`** — What you get back after calling `step()`. It contains:
- The new observation (for the next policy decision)
- The world state (for rendering)
- The intent state (for UI display)
- Any events that happened this tick
- What action the NPC took
- The reward and reward breakdown

**`Runtime`** — The orchestrator. It owns:
- A `World` instance
- An `IntentManager` instance
- An `ObservationBuilder` instance
- An `EventQueue` instance
- A `RewardCalculator` instance

When you call `step(action)`:
1. Pop all pending events from the queue
2. Handle each event (new instructions, cancellations, user movement)
3. Record NPC position before action
4. Apply the NPC action to the world via World.step()
5. Detect collision (if NPC tried to move but position unchanged)
6. Call IntentManager.update() to check completion/timeout
7. Calculate reward via RewardCalculator
8. Build the next observation via ObservationBuilder.build()
9. Package everything into a StepResult and return it

### How They Connect

```
External input → EventQueue (buffered)
                      ↓
Runtime.step(action) pops events, processes them
                      ↓
         ┌───────────┴───────────┐
         ↓                       ↓
   IntentManager            World.step()
   (new/cancel)             (apply action)
         ↓                       ↓
   IntentManager.update()  ←────┘
   (check completion)
         ↓
   RewardCalculator.calculate()
         ↓
   ObservationBuilder.build()
         ↓
   StepResult returned to caller
```

**The tick cycle (happens 16 times per second):**

1. **Process any pending events** — new instructions from user, movement commands, cancellations
2. **Let the policy choose an action** — given the current observation
3. **Apply the action to the world** — NPC moves, speaks, or waits
4. **Update the intent** — check if it's completed or timed out
5. **Calculate reward** — how well did the NPC do this tick?
6. **Build the next observation** — ready for the next tick

**What it provides:**
- `runtime.step(action)` — advance by one tick, get back the result with reward
- `runtime.submit_instruction(text, ...)` — queue a new instruction
- `runtime.cancel_instruction()` — cancel the current task
- `runtime.get_observation()` — get what the NPC currently sees
- `runtime.get_state()` — get everything (for debugging or Unity)

**Why it matters:** This is the glue. It coordinates the world, the intent system, the observation builder, and reward calculation into a single coherent loop that can be driven by a policy.

---

## Part 5: Reward (`src/reward/`)

**What it does:** Provides the learning signal that tells the NPC how well it's doing.

### Two-Phase Training Approach

Training uses a **phased reward strategy** to help the agent learn effectively:

**Phase 1: Exploration (No Penalties)**
```
Goal: Learn "moving toward target = good"
- No collision penalty (explore freely)
- No oscillation penalty
- No time pressure
- Only WAIT penalty (-0.1) to discourage inaction
- Bigger completion bonus (+15.0)
```

**Phase 2: Precision (Penalties Added)**
```
Goal: Learn efficient, smooth navigation
- Light collision penalty (-0.1)
- Light oscillation penalty (-0.05)
- Time pressure (-0.01/tick)
- Stronger WAIT penalty (-0.2)
```

The transition happens automatically when:
- Episode count reaches threshold (default: 2000), OR
- Success rate exceeds threshold (default: 50%)

This prevents the "frozen agent" problem where penalties kill exploration before the agent learns that movement is valuable.

### Classes and How They Work

**`RewardConfig`** — Holds all the reward weights. Everything is configurable so you can tune the balance between different incentives:
- `progress_scale` (5.0) — multiplier for distance-based progress
- `completion_bonus` (10.0) — reward for completing an intent
- `timeout_penalty` (-5.0) — penalty for failing to complete in time
- `cancel_penalty` (0.0) — penalty for user cancellation (not the NPC's fault)
- `time_penalty` (-0.01) — small penalty each tick to encourage speed
- `collision_penalty` (-0.1) — penalty for bumping into obstacles
- `oscillation_penalty` (-0.05) — penalty for back-and-forth movement

**Static factory methods:**
- `RewardConfig.exploration_phase()` — returns config for Phase 1
- `RewardConfig.precision_phase()` — returns config for Phase 2

**`RewardInfo`** — A breakdown of all reward components for a single tick. Useful for debugging and understanding what the NPC is being rewarded/penalized for.

**`RewardCalculator`** — The main class that computes rewards each tick. It:
- Tracks the previous distance to target (for progress calculation)
- Tracks recent actions (for oscillation detection)
- Combines all reward components into a total

When `calculate()` is called:
1. **Progress reward** — compare current distance to previous distance. Closer = positive, farther = negative
2. **Terminal rewards** — if intent just ended, apply completion bonus or timeout penalty
3. **Time penalty** — small negative if intent is active (encourages finishing quickly)
4. **Collision penalty** — if NPC tried to move but couldn't (hit something)
5. **Oscillation penalty** — if NPC reversed direction (e.g., RIGHT then LEFT)

### How It Connects

```
Runtime.step(action)
         ↓
Detects collision (position unchanged after move?)
         ↓
IntentManager.update() → intent_event if completed/timeout
         ↓
RewardCalculator.calculate(
    world_state,
    intent_state,
    action,
    collision_occurred,
    intent_event
)
         ↓
RewardInfo with breakdown → included in StepResult
```

**Reward components explained:**

| Component | When | Value | Purpose |
|-----------|------|-------|---------|
| Progress | Every tick | ±distance_delta | Encourage moving toward target |
| Completion | Intent completes | +10.0 | Big reward for success |
| Timeout | Intent times out | -5.0 | Penalty for taking too long |
| Time | Every tick (with intent) | -0.01 | Encourage finishing quickly |
| Collision | Hit obstacle | -0.1 | Discourage bumping into things |
| Oscillation | Reversed direction | -0.05 | Discourage useless wiggling |

**What it provides:**
- `calculator.calculate(...)` — compute reward for this tick
- `calculator.reset()` — reset internal state (call when episode/intent resets)
- `calculator.get_config()` — get current reward weights

**Why it matters:** Without rewards, the NPC has no idea what's good or bad. This module defines the incentive structure that shapes learning.

---

## Part 6: Training (`src/training/`)

**What it does:** Trains the NPC using Proximal Policy Optimization (PPO).

### Classes and How They Work

**`NPCEnv`** — A Gym-compatible environment wrapper around our Runtime. This makes our system work with standard RL libraries like stable-baselines3.

Episode structure:
1. `reset()` — create fresh world, pick random object, submit instruction like "Go to the red triangle"
2. `step(action)` — advance one tick, return observation, reward, done flag
3. Episode ends when: intent completes, intent times out, or max steps reached

Key properties:
- Action space: Discrete(5) — UP, DOWN, LEFT, RIGHT, WAIT (SPEAK excluded for MVP)
- Observation space: Box(575,) — the observation vector
- Automatically generates random instructions each episode

**`TrainerConfig`** — Holds all training hyperparameters:
- `total_timesteps` — how long to train
- `n_envs` — parallel environments for faster training
- `learning_rate`, `batch_size`, `n_epochs` — PPO parameters
- `gamma`, `gae_lambda` — discount and advantage estimation
- `ent_coef` — entropy bonus for exploration
- Save/log directories and frequencies

**`Trainer`** — The main training controller. It:
- Creates vectorized environments (multiple NPCEnvs running in parallel)
- Initializes the PPO model from stable-baselines3
- Runs training with checkpointing and evaluation callbacks
- Saves and loads trained models

### How It Connects

```
Trainer.setup()
      ↓
Creates n parallel NPCEnv instances
      ↓
Each NPCEnv wraps a Runtime
      ↓
PPO model initialized (MLP policy, ~82k parameters)
      ↓
Trainer.train()
      ↓
PPO collects rollouts across all envs
      ↓
Updates policy using clipped objective
      ↓
Repeats until total_timesteps reached
      ↓
Trainer.save() → saves model to disk
```

**Training flow:**

```
┌─────────────────────────────────────────────────────────────┐
│  For each update:                                           │
│                                                             │
│  1. Collect n_steps × n_envs transitions                   │
│     - Each env: obs → policy → action → step → reward      │
│                                                             │
│  2. Compute advantages using GAE                            │
│                                                             │
│  3. Update policy for n_epochs                              │
│     - Clipped surrogate objective (PPO)                     │
│     - Value function loss                                   │
│     - Entropy bonus                                         │
│                                                             │
│  4. Periodically evaluate and save checkpoints              │
└─────────────────────────────────────────────────────────────┘
```

**What it provides:**
- `trainer.setup()` — initialize environments and model
- `trainer.train()` — run the full training loop
- `trainer.save(name)` / `trainer.load(path)` — save/load models
- `trainer.predict(obs)` — get action from trained policy
- `trainer.evaluate(n_episodes)` — test current performance

**Why it matters:** This is where learning happens. PPO is a stable, well-tested algorithm that balances exploration and exploitation while keeping policy updates conservative.

---

## How It All Connects

**Intended Architecture:**
```
User says: "Go to the blue triangle"
              │
              ▼
        ┌─────────────┐
        │    LLM      │  (Future) Interprets language
        │   Layer     │  Outputs: NAVIGATE(target="blue triangle")
        └─────────────┘
              │
              ▼
        ┌─────────────┐
        │   Target    │  (Future) Resolves "blue triangle" → obj_7
        │  Resolver   │  Validates unambiguous
        └─────────────┘
              │
              ▼
        ┌─────────────┐
        │   Intent    │  Receives structured command
        │   Manager   │  Tracks: target_id=obj_7, action=NAVIGATE
        └─────────────┘
              │
              ▼
        ┌─────────────┐
        │ Observation │  Packages world state + goal info
        │   Builder   │
        └─────────────┘
              │
              ▼
        ┌─────────────┐
        │  RL Policy  │  Neural network picks action (0-4)
        │   (PPO)     │  Executes motor skills
        └─────────────┘
              │
              ▼
        ┌─────────────┐
        │    World    │  Applies action, updates positions
        └─────────────┘
              │
              ▼
        ┌─────────────┐
        │   Reward    │  Calculates learning signal
        └─────────────┘
              │
              ▼
        ┌─────────────┐
        │   Runtime   │  Orchestrates everything, loops back
        └─────────────┘
```

**Current MVP (Phase 1):** The LLM and Target Resolver layers are bypassed — the system directly provides `target_id` to the RL agent. This isolates RL training from language understanding.

---

## What's Built vs. What's Next

**Done (RL Execution Foundation):**
- World simulation with movement and collision
- Intent system with preemption and timeout
- Simplified observation builder (24 dims, no embeddings)
- Two-phase reward training (exploration → precision)
- Runtime loop orchestration
- Event system for user input
- Reward computation with configurable weights and phased configs
- PPO training pipeline with stable-baselines3
- Gym-compatible environment wrapper (`SimpleNPCEnv`)
- Curriculum learning for progressive difficulty (world size, success radius)
- Model saving/loading and evaluation
- Visualization tool for debugging agent behavior

**Current Focus (Phase 1 — RL Navigation):**
- Train robust navigation to structured targets
- Two-phase training: exploration (no penalties) → precision (penalties added)
- Curriculum from small worlds (8×8) to full size (64×64)
- Prove RL execution layer works reliably

**Next Priority (Phase 2 — Target Resolution):**
- Deterministic/supervised layer to resolve descriptions → target_id
- Validate ambiguity before passing to RL
- Extended action types (pick up, drop, etc.)

**Future (Phase 3 — LLM Integration):**
- LLM layer for language understanding and dialogue
- LLM → Target Resolver → RL Executor pipeline
- Complex multi-step instructions decomposed by LLM
- Unity/game engine integration

---

## Running the Demos

First, install dependencies:
```bash
pip install -r requirements.txt
```

Then run any demo:
```bash
python demo_world.py        # Test world simulation
python demo_intent.py       # Test intent + embeddings
python demo_observation.py  # Test observation building
python demo_runtime.py      # Test full runtime loop
python demo_reward.py       # Test reward computation
python demo_training.py     # Test training pipeline (short run)
```

---

## Training

### Simplified RL Executor (Recommended)

Uses `SimpleNPCEnv` with 24-dim observation (no language embeddings):

```bash
# Full training with two-phase rewards and curriculum
python train_simple.py --timesteps 1000000

# Customize phase transition
python train_simple.py --timesteps 1000000 \
    --exploration-episodes 3000 \
    --phase-success-threshold 0.6

# Disable phased rewards (use default penalties from start)
python train_simple.py --timesteps 1000000 --no-phased-rewards

# Disable curriculum learning
python train_simple.py --timesteps 1000000 --no-curriculum
```

### Legacy Training (with embeddings)

Uses `NPCEnv` with 575-dim observation:

```bash
python train.py --timesteps 500000
```

### Visualize Agent Behavior

```bash
# Watch trained simple agent
python visualize_agent.py --simple --model models/simple/best/best_model.zip

# Watch with greedy policy (for comparison)
python visualize_agent.py --simple --greedy

# Watch random baseline
python visualize_agent.py --simple --random
```

### Monitor with TensorBoard

```bash
tensorboard --logdir logs/simple
```

---

## Design Principles

1. **Python is the brain** — all logic lives here, the game engine just renders
2. **Separation of concerns** — LLM for language understanding, RL for physical execution
3. **RL as motor skill, not language parser** — the agent learns robust execution, not semantic interpretation
4. **Deterministic validation** — target resolution validates commands before execution; ambiguity is caught, not guessed
5. **Real-time, not episodic** — instructions can arrive anytime, even mid-task
6. **One intent at a time** — new instructions preempt old ones (queue possible in future)
7. **Deterministic completion** — success is measured by world state, not guesswork
8. **Engine-agnostic** — swap Unity for Godot or anything else without retraining
9. **Simple first, scale later** — start with navigation execution, add action types and LLM integration incrementally
