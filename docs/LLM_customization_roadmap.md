## LLM Customization Roadmap

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