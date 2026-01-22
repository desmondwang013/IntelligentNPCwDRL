The system is designed as a modular NPC toolkit where a large language model handles conversation and intent, while an action system executes reliable in-world behaviors through specialized modules. The key principle is separation of responsibilities: the LLM speaks to the user and reasons about intent, ambiguity, and feasibility, while action modules never speak directly to the user and never consume raw natural language. Instead, modules receive a strict structured command and return a strict structured result. This separation keeps the system stable, testable, and engine-agnostic.

At runtime, the workflow is: the User sends a natural-language instruction to the LLM, which produces a structured command in a constrained format (for example, JSON). That command does not go directly to any reinforcement learning policy. It first goes through an Execution Gateway layer, which is a deterministic boundary responsible for safety, grounding, and routing. The Execution Gateway validates the command against a schema, checks it against the current world snapshot, resolves references to concrete targets when possible, and decides whether the command is executable or requires clarification. If the command is invalid, unsafe, references nonexistent entities, or is ambiguous, the gateway blocks execution and returns a structured “needs clarification” result. The LLM then asks the user a precise follow-up question rather than sending an uncertain command to the action system. If the command is valid and grounded, the gateway routes it to the correct action module (Navigation, Combat, or other modules) through a stable internal interface.

Internally, the system is organized as a set of action modules behind the Execution Gateway. Each module is responsible for one skill category and can be implemented using reinforcement learning, deterministic logic, or a hybrid approach depending on the skill. The project direction is to avoid scripting everything, but also to avoid forcing reinforcement learning into places where deterministic logic is cleaner and more reliable. As a result, complex reactive “how” behaviors such as movement and avoidance are strong candidates for learned policies, while crisp atomic interactions can remain deterministic when appropriate. Regardless of implementation, every module must follow the same input and output contract so the system stays modular and easy to integrate into different games.

The Navigation Module can be implemented as a hybrid pipeline that combines deterministic planning with learned execution. Within the Navigation Module, the incoming request is treated as a structured navigation command rather than raw text. After the Execution Gateway has validated the command and resolved the target, the Navigation Module performs deterministic global planning to produce a concrete route, typically represented as a sequence of waypoints. A deterministic pathfinder (such as a grid-based planner) uses the current walkability representation of the world to compute these waypoints. The module then hands the waypoint sequence to an RL Executor which is responsible for local movement execution. The RL Executor does not decide what the goal is. It is trained to move the agent robustly toward the next waypoint, handle local motion details, and recover from minor local disturbances. The Navigation Module monitors progress and can trigger replanning when progress stalls or when the world changes enough to invalidate the current route. The module returns a structured result to the Execution Gateway, including success or failure status, progress metrics such as steps used and final distance, and failure reasons such as timeout or blockage. The LLM uses these results to narrate outcomes, ask clarifying questions, or propose alternative actions to the user.

As the project expands beyond navigation, the same pattern applies. The system supports multiple modules such as Navigation, Combat, and other future skills, but they remain coordinated through the same gateway and contract. Combat behavior should not reinvent locomotion separately from navigation. Instead, combat logic should express tactical intent and constraints, while the movement execution remains consistent through the navigation and locomotion stack. This avoids brittle mode switching and ensures the NPC maintains a coherent movement style across tasks. For example, a combat module can request subgoals such as “maintain distance,” “seek cover,” or “approach to range,” while navigation and locomotion execute the resulting movement safely and consistently.

The Execution Gateway is the missing layer that makes this architecture practical. It ensures the LLM can be flexible in conversation while the action system remains strict and reliable. It also ensures that module routing is deterministic and testable instead of relying on the LLM to directly select internal modules. This gateway is the system’s “narrow waist” that enforces consistency: the LLM can change, modules can evolve, and engines can vary, but the command and result contracts remain stable. This stability is essential for building an engine-agnostic toolkit that developers can integrate without training reinforcement learning from scratch for each game.


Workflow
  U[User] --> LLM[LLM: Intent Manager]
  LLM --> CMD[Structured Command (JSON)]
  CMD --> GW[Execution Gateway\nValidator + Resolver + Router]

  GW --> NAV[Navigation Module]
  GW --> COM[Combat Module]
  GW --> OTH[Other Modules...]

  NAV --> RES[Structured Result]
  COM --> RES
  OTH --> RES

  RES --> GW
  GW --> LLM
  LLM --> U

  subgraph NAV[Navigation Module]
    IN[Incoming Command] --> GW2[Deterministic Validator + Resolver\nReturn "needs clarification" if unclear]
    GW2 --> TID[target_id]
    TID --> PF[Deterministic Pathfinder]
    PF --> WPS[Waypoints]
    WPS --> RL[RL Executor (Local Control)]
    RL --> OUT[Navigation Result\n(success/fail + metrics)]
  end

  LLM[LLM: Intent Manager] --> IN
  OUT --> LLM
  LLM --> U[User]

