# Navigation Approaches for Complex Environments

This document outlines two approaches for enabling the RL-based NPC to navigate complex game environments with walls, corridors, mazes, and dense obstacles.

## Current System Limitations

The current RL agent works well for:
- Open environments with scattered obstacles
- Short-to-medium distance navigation
- Local obstacle avoidance

However, it struggles with:
- Maze-like structures requiring global planning
- Dead ends and backtracking
- Environments where the direct path is blocked by walls
- Complex room-and-corridor layouts

**Root cause**: The agent only observes 5 nearest obstacles and has no global map awareness. It cannot plan paths that require temporarily moving away from the target.

---

## Option 1: Hybrid Approach (A* Pathfinding + RL Motor Control)

### Overview

Combine classical pathfinding for global route planning with RL for local movement execution.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Game World                              │
│                                                                 │
│   ┌────────────┐      ┌────────────┐      ┌─────────────────┐  │
│   │    A*      │ ---> │  Waypoint  │ ---> │   RL Motor      │  │
│   │ Pathfinder │      │   Queue    │      │   Controller    │  │
│   └────────────┘      └────────────┘      └─────────────────┘  │
│                                                                 │
│   Responsibility:      Responsibility:     Responsibility:      │
│   - Global path        - Store path        - Move between       │
│   - Obstacle map       - Track progress      waypoints          │
│   - Guaranteed         - Next target       - Smooth motion      │
│     solution                               - Local avoidance    │
└─────────────────────────────────────────────────────────────────┘
```

### A* Pathfinding Component

**What it is**: A classical graph search algorithm that finds the shortest path between two points while avoiding obstacles.

**Input**:
- Start position (NPC location)
- Goal position (target location)
- Obstacle map (grid representation of the world)

**Output**:
- Ordered list of waypoints from start to goal
- Returns empty if no path exists

**Characteristics**:
- Deterministic (same input = same output)
- Guaranteed optimal (finds shortest path)
- Fast (typically milliseconds even for large maps)
- Well-understood (invented 1968, used in every game)

**Pseudocode**:
```python
def a_star(start, goal, obstacle_grid):
    open_set = PriorityQueue()
    open_set.put(start, priority=0)
    came_from = {}
    cost_so_far = {start: 0}

    while not open_set.empty():
        current = open_set.get()

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, obstacle_grid):
            new_cost = cost_so_far[current] + distance(current, neighbor)

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                open_set.put(neighbor, priority)
                came_from[neighbor] = current

    return []  # No path found
```

### RL Motor Controller (Current System)

**Role in hybrid system**: Navigate between consecutive waypoints.

**Changes needed**: Minimal
- Target becomes "next waypoint" instead of "final target"
- When waypoint reached, pop next from queue
- Continue until final target reached

**Why RL is still valuable**:
- Smooth, natural-looking movement
- Handles minor dynamic obstacles
- Learned behavior looks less "robotic" than pure pathfinding
- Can incorporate character-specific movement styles

### Integration Architecture

```python
class HybridNavigator:
    def __init__(self, pathfinder, rl_agent):
        self.pathfinder = pathfinder  # A* implementation
        self.rl_agent = rl_agent      # Trained PPO model
        self.waypoints = []
        self.current_waypoint_idx = 0

    def set_goal(self, start_pos, goal_pos, obstacle_map):
        """Compute path and initialize waypoint queue."""
        self.waypoints = self.pathfinder.find_path(
            start_pos, goal_pos, obstacle_map
        )
        self.current_waypoint_idx = 0

    def get_action(self, observation):
        """Get next action from RL agent toward current waypoint."""
        if self.current_waypoint_idx >= len(self.waypoints):
            return WAIT  # Goal reached

        current_waypoint = self.waypoints[self.current_waypoint_idx]

        # Modify observation to target current waypoint
        obs_with_waypoint = self._set_target(observation, current_waypoint)

        # Get action from RL agent
        action = self.rl_agent.predict(obs_with_waypoint)

        return action

    def update(self, npc_position):
        """Check if current waypoint reached, advance if so."""
        if self.current_waypoint_idx >= len(self.waypoints):
            return

        current_waypoint = self.waypoints[self.current_waypoint_idx]
        distance = compute_distance(npc_position, current_waypoint)

        if distance < WAYPOINT_THRESHOLD:
            self.current_waypoint_idx += 1
```

### Obstacle Map Representation

For A* to work, you need a grid representation of obstacles:

```python
class ObstacleMap:
    def __init__(self, world_size, resolution=1.0):
        self.resolution = resolution
        self.grid_size = int(world_size / resolution)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)

    def add_obstacle(self, position, radius):
        """Mark cells occupied by an obstacle."""
        gx, gy = self._world_to_grid(position)
        grid_radius = int(radius / self.resolution) + 1

        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if dx*dx + dy*dy <= grid_radius*grid_radius:
                        self.grid[nx, ny] = True

    def is_blocked(self, grid_x, grid_y):
        """Check if a grid cell is blocked."""
        return self.grid[grid_x, grid_y]
```

### Pros and Cons

**Pros**:
- Guaranteed to find path if one exists
- Handles arbitrarily complex environments
- Your current RL agent works as-is (just change target to waypoint)
- Industry-proven approach (used in almost every commercial game)
- Easy to debug (can visualize planned path)
- Fast to implement (~100-200 lines for A*)

**Cons**:
- Requires maintaining obstacle map (sync with world state)
- Raw A* paths can look "robotic" (fixable with path smoothing)
- Two systems to maintain
- Not "pure learning" (hybrid classical + ML)

### Recommended Resources

- [A* Pathfinding Tutorial (Red Blob Games)](https://www.redblobgames.com/pathfinding/a-star/introduction.html) - Excellent visual explanation
- [Navigation Meshes (Unity)](https://docs.unity3d.com/Manual/nav-NavigationSystem.html) - Industry-standard for 3D games
- Python libraries: `python-pathfinding`, `networkx`

---

## Option 4: Hierarchical Reinforcement Learning

### Overview

Use two levels of learned policies: a high-level "manager" that sets subgoals and a low-level "worker" that executes them.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Hierarchical RL System                        │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              High-Level Policy (Manager)                │  │
│   │                                                         │  │
│   │  Observation: Larger spatial view (e.g., 16x16 grid)    │  │
│   │  Action: Select subgoal position for low-level          │  │
│   │  Runs: Every K steps (e.g., K=25)                       │  │
│   │  Reward: Sparse (final goal reached)                    │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              │ Subgoal                          │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Low-Level Policy (Worker)                  │  │
│   │                                                         │  │
│   │  Observation: Local view + current subgoal              │  │
│   │  Action: Primitive movement (UP/DOWN/LEFT/RIGHT/WAIT)   │  │
│   │  Runs: Every step                                       │  │
│   │  Reward: Dense (progress toward subgoal)                │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              │ Movement                         │
│                              ▼                                  │
│                        Environment                              │
└─────────────────────────────────────────────────────────────────┘
```

### High-Level Policy (Manager)

**Role**: Strategic navigation decisions

**Observation space** (example):
- Local grid map (16x16 cells around NPC showing obstacles)
- NPC position (normalized)
- Final goal position (normalized)
- Previous subgoal success/failure

**Action space** (options):
- **Continuous**: Output (x, y) offset for subgoal position
- **Discrete**: Select from predefined directions/distances (e.g., 8 directions × 3 distances = 24 actions)
- **Grid-based**: Select a cell in local grid as subgoal

**Temporal abstraction**: Runs every K environment steps (e.g., K=10-50)
- Reduces decision complexity
- Allows low-level to execute before reassessment
- If subgoal reached early, can trigger new decision

**Reward**:
- Primary: Sparse reward when final goal reached
- Optional: Small reward for subgoals that reduce distance to goal
- Penalty: For selecting unreachable subgoals

### Low-Level Policy (Worker)

**Role**: Motor execution toward current subgoal

**This is essentially your current RL agent** with one change:
- Target = current subgoal (not final target)
- When subgoal reached, high-level provides new subgoal

**Observation space**: Current 30-dim observation, but target_position = subgoal

**Action space**: Same as current (UP, DOWN, LEFT, RIGHT, WAIT)

**Reward**: Same as current (progress toward target, completion bonus)

### Training Approaches

#### Approach A: Feudal Networks / HIRO

Train both levels simultaneously with:
- High-level rewarded for final goal
- Low-level rewarded for reaching subgoals
- Gradients flow through both networks

**Challenge**: Non-stationary - low-level changes affect high-level's optimal strategy

```python
# Simplified HIRO-style training loop
for episode in episodes:
    state = env.reset()
    high_level_state = get_high_level_obs(state)
    subgoal = high_level_policy.select_action(high_level_state)

    for step in range(max_steps):
        # Low-level acts toward subgoal
        low_level_obs = combine(state, subgoal)
        action = low_level_policy.select_action(low_level_obs)

        next_state, reward, done = env.step(action)

        # Low-level reward: progress toward subgoal
        low_level_reward = compute_subgoal_reward(state, next_state, subgoal)
        low_level_buffer.add(low_level_obs, action, low_level_reward)

        # Every K steps or subgoal reached: high-level decision
        if step % K == 0 or subgoal_reached(next_state, subgoal):
            high_level_reward = reward  # Environment reward
            high_level_buffer.add(high_level_state, subgoal, high_level_reward)

            high_level_state = get_high_level_obs(next_state)
            subgoal = high_level_policy.select_action(high_level_state)

        state = next_state
```

#### Approach B: Pre-train Low-Level, Then Train High-Level

1. First, train low-level on simple point-to-point navigation (already done!)
2. Freeze low-level policy
3. Train high-level to select good subgoals for frozen low-level

**Advantage**: Simpler, low-level is stable
**Disadvantage**: Low-level can't adapt to high-level's subgoal style

#### Approach C: Options Framework

Define discrete "options" (temporally extended actions):
- Option "go north": Low-level moves north until blocked or K steps
- Option "go east": Low-level moves east until blocked or K steps
- etc.

High-level learns which options to invoke.

### Observation: Local Grid Map

Hierarchical RL typically needs richer spatial observation:

```python
def build_local_grid_observation(world_state, npc_pos, grid_size=16):
    """Build a local grid map around NPC showing obstacles."""
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    half_size = grid_size // 2
    cell_size = 1.0  # world units per cell

    for obj in world_state["objects"]:
        obj_pos = obj["position"]
        # Convert to grid coordinates relative to NPC
        rel_x = int((obj_pos["x"] - npc_pos["x"]) / cell_size) + half_size
        rel_y = int((obj_pos["y"] - npc_pos["y"]) / cell_size) + half_size

        if 0 <= rel_x < grid_size and 0 <= rel_y < grid_size:
            grid[rel_y, rel_x] = 1.0  # Mark as obstacle

    return grid  # Shape: (16, 16), can flatten or use CNN
```

### Network Architecture

**High-Level Policy** (needs spatial understanding):
```python
class HighLevelPolicy(nn.Module):
    def __init__(self, grid_size=16, num_subgoal_actions=24):
        super().__init__()
        # CNN for processing local grid map
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # MLP for goal information
        self.goal_mlp = nn.Sequential(
            nn.Linear(4, 32),  # NPC pos (2) + goal pos (2)
            nn.ReLU(),
        )
        # Combined decision
        self.policy_head = nn.Sequential(
            nn.Linear(64 * grid_size * grid_size + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_subgoal_actions),
        )

    def forward(self, grid_map, goal_info):
        grid_features = self.conv(grid_map)
        goal_features = self.goal_mlp(goal_info)
        combined = torch.cat([grid_features, goal_features], dim=1)
        return self.policy_head(combined)
```

**Low-Level Policy**: Your current 64-64 MLP architecture works fine.

### Pros and Cons

**Pros**:
- Fully learned (no hand-coded algorithms)
- Can discover creative/emergent solutions
- End-to-end differentiable
- More flexible than fixed pathfinding
- Academically interesting (publishable research)

**Cons**:
- Significantly harder to train
- Two coupled policies create training instability
- May not converge or may find suboptimal paths
- Requires more complex observation (local grid)
- Needs careful reward engineering
- Less interpretable than A*
- No guarantees (might fail on some maps)

### Recommended Resources

**Papers**:
- [HIRO: Hierarchical RL with Off-Policy Correction](https://arxiv.org/abs/1805.08296) (Nachum et al., 2018)
- [FeUdal Networks](https://arxiv.org/abs/1703.01161) (Vezhnevets et al., 2017)
- [Option-Critic Architecture](https://arxiv.org/abs/1609.05140) (Bacon et al., 2017)
- [HAM: Hierarchies of Abstract Machines](https://people.eecs.berkeley.edu/~parr/ham.pdf) (Parr & Russell, 1998)

**Tutorials**:
- [Hierarchical RL Tutorial (Spinning Up)](https://spinningup.openai.com/en/latest/)
- [Stable-Baselines3 Custom Policies](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)

---

## Comparison Summary

| Aspect | Hybrid (A* + RL) | Hierarchical RL |
|--------|------------------|-----------------|
| **Implementation Time** | ~1 week | ~1 month+ |
| **Training Complexity** | Low (only low-level) | High (two coupled policies) |
| **Reliability** | High (A* guarantees) | Uncertain (may not converge) |
| **Generalization** | Any map with obstacles | Only trained scenarios |
| **Interpretability** | High (can visualize path) | Low (learned behaviors) |
| **Industry Usage** | Very common | Mostly research |
| **Academic Interest** | Lower | Higher |
| **Reuse of Current RL** | Direct (as motor controller) | Direct (as low-level policy) |
| **Handles Mazes** | Yes (guaranteed) | Maybe (if trained well) |
| **Handles Dynamic Obstacles** | Needs re-planning | Can adapt (if trained) |

---

## Recommendation

**For a practical game project**: Start with **Hybrid (A* + RL)**
- Your RL agent already handles motor control well
- A* is simple to implement and guaranteed to work
- Industry-proven, debuggable, maintainable

**For research exploration**: Consider **Hierarchical RL**
- After hybrid system is working
- As an experimental alternative
- Good for publications/learning

**Suggested path forward**:
1. Implement Hybrid system first (get navigation working in complex maps)
2. Build Target Resolver and LLM integration
3. Optionally explore Hierarchical RL as research extension
