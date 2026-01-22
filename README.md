# Intelligent NPC with Deep Reinforcement Learning

A modular NPC toolkit where a large language model handles conversation and intent, while an action system executes reliable in-world behaviors through specialized modules.

---

## Core Principle

**Separation of responsibilities:**
- **LLM** speaks to the user, classifies intent, extracts task information
- **NPCController** validates, resolves targets, routes commands, and manages task state
- **Action Modules** execute behaviors, return structured results

The LLM never accesses world state directly. Action modules never speak to users. The Controller is the "narrow waist" that enforces contracts between them.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  USER                                                           │
│  "Go to the red circle" | "Good morning" | "Attack the enemy"  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLM Layer (Intent Parser)                                      │
│                                                                 │
│  Classifies input:                                              │
│  - TASK: extract action + raw target description                │
│  - CONVERSATION: respond naturally                              │
│  - UNSUPPORTED: flag impossible tasks                           │
│                                                                 │
│  Does NOT:                                                      │
│  - Know what objects exist in world                             │
│  - Resolve "red circle" to entity_id                            │
│  - Check if target is valid or ambiguous                        │
│                                                                 │
│  Examples:                                                      │
│  "go to the red circle" → {type: task, action: navigate,        │
│                            target: "red circle"}                │
│  "good morning"         → {type: conversation,                  │
│                            response: "Good morning!"}           │
│  "attack the enemy"     → {type: task, action: combat,          │
│                            target: "enemy"} [when Combat ready] │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (tasks only)
┌─────────────────────────────────────────────────────────────────┐
│  NPCController (Validator + Resolver + Router)                  │
│                                                                 │
│  - Validates action is supported                                │
│  - Resolves "red circle" → obj_3 using world state              │
│  - Checks target exists and is unambiguous                      │
│  - Routes to correct module (Navigation, Combat, etc.)          │
│  - Returns structured result or clarification request           │
│                                                                 │
│  Possible results:                                              │
│  - success: target resolved, routed to module                   │
│  - ambiguous: multiple matches, ask user to clarify             │
│  - not_found: target doesn't exist                              │
│  - unsupported: action not available                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Action Modules                                                 │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Navigation Module (Hybrid: A* + RL)                    │    │
│  │  - Receives: target_id (resolved by Gateway)            │    │
│  │  - A* pathfinder computes waypoints                     │    │
│  │  - RL Executor follows waypoints (local control)        │    │
│  │  - Returns: success/fail + metrics                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  [Next] Combat Module                                   │    │
│  │  - Uses Navigation for approach/retreat                 │    │
│  │  - Attack, defend, flee behaviors                       │    │
│  │  - Tactical decisions based on health/distance          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  [Future] Inventory Module                              │    │
│  │  - Pick up, drop, use items                             │    │
│  │  - Interact with containers                             │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Structured Result → Controller → LLM → User                    │
│                                                                 │
│  LLM narrates outcome:                                          │
│  - success → "Done, I'm at the red circle."                     │
│  - ambiguous → "Which one? I see a red circle and red square."  │
│  - not_found → "I don't see that here."                         │
│  - unsupported → "Sorry, I can't do that."                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## World

The simulation environment where the NPC operates.

**Entities:**
- **NPC** - the agent controlled by the system
- **User** - the human player giving instructions
- **WorldObjects** - interactive objects with attributes:
  - Color: red, blue, green, yellow, purple, orange
  - Shape: circle, triangle, square, diamond
  - Size: small, medium, large

**Actions (RL Executor):**
- `MOVE_UP`, `MOVE_DOWN`, `MOVE_LEFT`, `MOVE_RIGHT` - movement
- `WAIT` - stay in place

**World State:**
- 64×64 grid (configurable)
- Collision detection
- Objects and entities have collision radii

---

## LLM Layer

**Role:** Language interface between user and system.

**Responsibilities:**
1. Classify input as TASK or CONVERSATION
2. For tasks: extract action + raw target description
3. For conversation: respond naturally
4. For unsupported tasks: flag appropriately

**Does NOT:**
- Access world state
- Resolve target descriptions to entity IDs
- Know what objects exist

**Output Format:**
```json
// Task
{"type": "task", "action": "navigate", "target": "red circle"}

// Conversation
{"type": "conversation", "response": "Hello! How can I help?"}

// Unsupported
{"type": "task", "action": "unsupported"}
```

**Supported Actions:**
- `navigate` - move to a target
- `wait` - stay in place
- `unsupported` - recognized as task but cannot perform

**Current Implementation:**
- Backend: Ollama with Qwen3:4b
- Thinking mode disabled for fast responses (~1s)
- JSON output mode for structured responses
- Alternate backend: llama-cpp-python (optional)

---

## NPCController

**Role:** The "narrow waist" between LLM and action modules.

**Responsibilities:**
1. Validate command schema
2. Resolve target descriptions to entity_ids using world state
3. Check for ambiguity (multiple matches)
4. Check target exists
5. Route to appropriate module
6. Return structured results

**Resolution Example:**
```
Input: {"action": "navigate", "target": "red circle"}
World state: [obj_1: red circle, obj_2: blue square, obj_3: red triangle]

Controller resolves "red circle" → finds obj_1 → routes to Navigation

If "red" specified without shape:
→ multiple matches (obj_1, obj_3) → return ambiguous
```

**Status:** Implemented and actively used (see `src/controller/`).

---

## Navigation Module

**Architecture:** Hybrid (A* pathfinding + RL execution)

```
target_id → A* Pathfinder → Waypoints → RL Executor → Result
```

**Components:**

1. **Obstacle Map** (`src/navigation/obstacle_map.py`)
   - Grid-based representation of walkable/blocked cells
   - Built from world state

2. **A* Pathfinder** (`src/navigation/pathfinding.py`)
   - Computes global path as waypoints
   - Path smoothing via line-of-sight checks

3. **Hybrid Navigator** (`src/navigation/hybrid_navigator.py`)
   - Manages waypoint progression
   - Triggers replanning when stuck or world changes

4. **RL Executor** (trained PPO model)
   - Follows waypoints locally
   - Handles obstacle avoidance
   - Robust to minor disturbances

**Observation Space (30 dimensions):**

| Component | Dimensions | Description |
|-----------|------------|-------------|
| NPC position | 2 | Normalized (x, y) |
| Target position | 2 | Current waypoint or final goal |
| Direction to target | 2 | Unit vector |
| Distance to target | 1 | Normalized |
| Target reached flag | 1 | 1.0 if within threshold |
| Nearest obstacles | 15 | 5 × (relative x, y, radius) |
| Intent age | 1 | Time pressure |
| Previous action | 5 | One-hot encoded (UP/DOWN/LEFT/RIGHT/WAIT) |
| Action blocked | 1 | 1.0 if last move was blocked by collision |

---

## Training

**Two-Phase Reward Strategy:**

**Phase 1: Exploration**
- No collision/oscillation penalties
- Agent learns "moving toward target = good"
- Large completion bonus

**Phase 2: Precision**
- Collision and oscillation penalties added
- Time pressure
- Efficient, smooth navigation

**Curriculum Learning:**
- Start with small worlds (8×8)
- Progress to full size (64×64)
- Gradually increase object count

**Run Training:**
```bash
python train_simple.py --timesteps 1000000
```

**Visualize:**
```bash
python visualize_agent.py --model models/simple/final_model.zip
python visualize_hybrid.py --world-size 16 --num-objects 20  # Test A* + RL
```

---

## Project Structure

```
src/
├── controller/         # NPCController (validation, resolution, routing)
│   ├── npc_controller.py   # Main controller class
│   ├── world_query.py      # Read-only world state queries
│   ├── intent.py           # Intent data structures
│   └── intent_manager.py   # Intent lifecycle tracking
├── llm/                # LLM integration
│   ├── client.py       # Ollama/llama-cpp backends
│   ├── intent_parser.py # Parse user input to structured commands
│   └── grammars.py     # JSON output constraints (for future use)
├── navigation/         # Hybrid navigation system
│   ├── base.py         # Navigator interface
│   ├── obstacle_map.py # Grid-based obstacles
│   ├── pathfinding.py  # A* algorithm
│   └── hybrid_navigator.py # A* + waypoint management
├── world/              # Game world simulation
├── observation/        # Observation builder (30-dim)
├── reward/             # Phased reward system
├── runtime/            # Event system and runtime utilities
├── intent/             # Re-exports from controller (backward compat)
└── training/           # PPO training pipeline

docs/
└── project_scope_Jan19th.md  # Architecture reference

# Main scripts
train_simple.py         # Train navigation RL agent
demo_full_pipeline.py   # Full integration demo (LLM → Controller → Nav → Result)
test_controller.py      # Test NPCController
test_llm.py             # Test LLM integration
visualize_agent.py      # Watch RL agent navigate
visualize_hybrid.py     # Test A* + RL hybrid navigation
```

---

## Current State

### Completed

| Component | Status | Description |
|-----------|--------|-------------|
| World Simulation | ✅ Done | Collision detection, entity management, 16-64 grid |
| Navigation RL | ✅ Done | PPO-trained agent follows waypoints |
| Hybrid Navigation | ✅ Done | A* pathfinding + RL motor control |
| LLM Integration | ✅ Done | Ollama + Qwen3:4b, intent parsing |
| NPCController | ✅ Done | Validation, target resolution, routing |
| Full Pipeline | ✅ Done | User → LLM → Controller → Navigation → Result → User |

### What Works Now

```bash
# Run the full pipeline demo
python demo_full_pipeline.py --interactive

# Example interactions:
"go to the large purple circle"  → Navigates successfully
"walk to the player"             → Navigates to user
"go to the red thing"            → Returns ambiguous (multiple matches)
"wait here"                      → Immediate success
"attack the enemy"               → Returns unsupported
"hello there"                    → Conversation response
```

### Known Limitations (Current)

- LLM may misparse vague terms ("stuff" vs "thing")
- Conversation mode has no world awareness
- Only one action module (Navigation)

### Next Steps

| Priority | Task | Description |
|----------|------|-------------|
| 1 | **Combat Module** | Attack, defend, flee behaviors using Navigation |
| 2 | Inventory Module | Pick up, drop, use items |
| 3 | World-Aware Conversation | NPC can discuss visible objects |
| 4 | LLM Fine-tuning | Train on all action types (deferred until modules ready) |

---

## LLM Customization Roadmap (Future)

*Deferred until action modules (Combat, Inventory) are complete.*

### The Problem

Prompt engineering has limits. The LLM may:
- Parse "yellow stuff" as `shape: "stuff"` instead of `shape: null`
- Handle "thing" correctly but fail on "thingy", "object", "item"
- Lack world awareness in conversation mode

As game complexity grows, prompts become unwieldy and brittle.

### The Solution: Fine-tuning + Configuration

**Two-tier architecture that preserves engine-agnostic goals:**

```
┌─────────────────────────────────────────────────────────────┐
│  Base Fine-tuned Model (ships with toolkit)                 │
│  - Understands intent structure                             │
│  - Maps vague language → null fields                        │
│  - Knows abstract actions: MOVE, INTERACT, WAIT, COMBAT     │
│  - Outputs clean JSON                                       │
│  - Train once, use everywhere                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Game Config (JSON/YAML, NO retraining needed)              │
│  - Action mapping: MOVE → "walk", "run", "teleport"         │
│  - Object types: "enemy", "chest", "door"                   │
│  - Attribute vocabulary: colors, sizes, custom properties   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  NPCController (translates abstract → game-specific)        │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** Fine-tune on *abstract patterns*, not game-specific vocabulary.

### Training Data Strategy

Generate diverse (input, output) pairs covering:

1. **Vague language variations:**
   ```
   "go to the yellow stuff" → {action: "move", color: "yellow", shape: null}
   "walk to that thingy"    → {action: "move", target_type: "object"}
   "head over there"        → {action: "move", target_type: "location"}
   ```

2. **Abstract action types:**
   - MOVE: walk, go, run, head, travel, approach
   - INTERACT: talk, use, open, activate, pick up
   - COMBAT: attack, fight, strike, defend
   - WAIT: wait, stay, hold, stop

3. **Edge cases:**
   - Typos, slang, incomplete sentences
   - Ambiguous references ("that one", "the other")
   - Multi-intent ("go there and pick it up")

### Implementation Phases

**Phase 1: Training Data Generator**
- Script to generate diverse training examples
- Bootstrap with current LLM + manual curation
- Target: 500-1000 high-quality examples

**Phase 2: QLoRA Fine-tuning**
- Efficient fine-tuning (~1-5% of parameters)
- Runs on consumer GPU (8-12GB VRAM)
- Training time: 30-60 minutes for 1000 samples

**Phase 3: World-Aware Conversation**
- RAG-style context injection for conversation mode
- NPC can discuss visible objects, recent actions
- No retraining needed, just context management

**Phase 4 (Optional): Game-Specific LoRA Adapters**
- For games needing specialized vocabulary
- Small adapter files (~50MB)
- Community sharing of game adapters

### Preserving Engine-Agnostic Goals

| Concern | Solution |
|---------|----------|
| Retraining per game | Abstract actions + config file |
| Deployment complexity | Ship pre-trained model |
| Novel game mechanics | Optional LoRA adapter |
| Hardware requirements | QLoRA works on 8GB VRAM |

**Most games need zero retraining** - just a YAML config defining their vocabulary.

---

## Legacy / Notes

- Older embedding-based intent components remain for reference and testing.
- These are not part of the current production flow and may be removed later.

---

## Design Principles

1. **Separation of concerns** - LLM speaks, Gateway validates, Modules execute
2. **LLM doesn't see world state** - Only receives structured results
3. **Gateway is deterministic** - Testable validation and resolution
4. **RL for motor control only** - No language parsing in RL
5. **Engine-agnostic** - Python is the brain, game engine just renders
6. **Modular contracts** - Stable interfaces between layers
7. **Simple first** - Navigation working, then add complexity

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Pull LLM model (requires Ollama installed)
ollama pull qwen3:4b

# Run the full pipeline demo (recommended)
python demo_full_pipeline.py --interactive

# Or run automated test sequence
python demo_full_pipeline.py

# Train navigation agent (if needed)
python train_simple.py --timesteps 500000

# Watch trained agent navigate
python visualize_agent.py --model models/simple/final_model.zip
```
