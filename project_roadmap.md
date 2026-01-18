# Project Roadmap (Revised)

## Phase 0: Navigation Reliability (Current)
- Goal: RL executor navigates smoothly and reliably to a target.
- Exit criteria:
  - Success rate consistently above target threshold across curriculum sizes.
  - Low collision/oscillation; reasonable episode length.

---

## Phase 1: LLM Integration (Core)

### 1. Local LLM Deployment
- Select local LLM (e.g., Qwen) and set up inference.
- Define latency and throughput expectations.

### 2. ActionPlan Interface (LLM → RL)
- Define strict JSON schema for ActionPlan.
- Include:
  - action_type (NAVIGATE, PICK_UP, DROP, etc.)
  - target descriptor (text)
  - optional params (destination, object constraints)
- Add deterministic validation and rejection handling.

### 3. Target Resolution
- Implement deterministic/supervised resolver:
  - description → target_id
  - ambiguity handling (if multiple candidates)
- Unit tests for object resolution accuracy.

### 4. End-to-End Intent Flow Test
- Test pipeline with controlled prompts:
  - User text → LLM → ActionPlan → resolver → target_id → RL
- Exit criteria:
  - High accuracy of ActionPlan parsing
  - Correct target resolution rate
  - RL completes navigation with high success rate

---

## Phase 2: Expanded Action Space
- Add interaction actions (pick up, drop, carry, attack, etc.).
- Extend resolver + validator to support new actions.
- Retrain RL executor for each new action type.

---

## Phase 3: Optional Semantic Audit Layer
- Add secondary LLM or VLM for semantic checks.
- Role: validate that observed behavior matches instruction intent.
- Only introduce after core pipeline is stable.

---

## Key Principles
- Keep RL focused on execution (motor skills only).
- Keep language understanding in LLM + deterministic resolver.
- Use strict schemas and automated tests to reduce integration drift.
