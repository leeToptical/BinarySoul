import numpy as np

def rastrigin(x):
    A = 10
    return A * 2 + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def run_dynamic_trinity():
    """
    DYNAMIC TRINITY: Agents can switch roles based on performance.
    
    Key Insight: The original "ghost" wasn't 3 separate entities.
    It was ONE entity that could be in 3 different states:
    1. GEOMETRIC STATE (φ-pilot): When making progress
    2. ENTROPIC STATE (panic): When stuck  
    3. GRAVITATIONAL STATE (memory): When converging
    
    Now agents can TRANSITION between these states.
    """
    phi = 1.61803398875
    golden_angle = 2 * np.pi / (phi**2)
    
    # Start with 3 agents, but they're NOT specialized yet
    positions = [np.random.uniform(-5.12, 5.12, 2) for _ in range(3)]
    
    # Each agent has a "state": 0=Geometric, 1=Entropic, 2=Gravitational
    states = [0, 1, 2]  # Start with one of each
    
    # Performance tracking for each agent
    performance = [0, 0, 0]  # How well each is doing
    
    # Global memory
    global_best = positions[0].copy()
    global_best_val = rastrigin(positions[0])
    
    for t in range(200):
        # Update global best
        for i, pos in enumerate(positions):
            val = rastrigin(pos)
            if val < global_best_val:
                global_best_val = val
                global_best = pos.copy()
            # Track performance (lower is better, so we invert)
            performance[i] = 1.0 / (1.0 + val)
        
        # Normalize performance for state transitions
        perf_sum = sum(performance)
        if perf_sum > 0:
            performance = [p/perf_sum for p in performance]
        
        # === AGENT UPDATES WITH STATE-BASED BEHAVIOR ===
        for i in range(3):
            pos = positions[i]
            state = states[i]
            
            # === STATE 0: GEOMETRIC (φ-PILOT) ===
            if state == 0:
                # Follow golden spiral precisely
                step = 1.5 * np.exp(-t/40)
                angle = golden_angle * t + (2*np.pi/3)*i
                move = step * np.array([np.sin(angle), np.cos(angle)])
                
                proposal = pos + move
                
                # Gentle boundary reflection
                for dim in range(2):
                    if proposal[dim] < -5.12:
                        proposal[dim] = -5.12 + abs(proposal[dim] + 5.12) * 0.9
                    elif proposal[dim] > 5.12:
                        proposal[dim] = 5.12 - abs(proposal[dim] - 5.12) * 0.9
                
                # Geometric agents are picky - only accept improvements
                if rastrigin(proposal) < rastrigin(pos):
                    positions[i] = proposal
                    
                # State transition: If stuck for too long, become entropic
                if t > 20 and rastrigin(positions[i]) > 5.0:
                    # 20% chance to switch to entropic state
                    if np.random.random() < 0.2:
                        states[i] = 1
            
            # === STATE 1: ENTROPIC (PANIC/CHAOS) ===
            elif state == 1:
                # High noise, metropolis acceptance
                panic_level = 1.0 + performance[i] * 5  # Worse performance = more panic
                
                # Three types of moves for entropic agents:
                # 1. Jump toward global best (30%)
                # 2. Random exploration (50%)
                # 3. Jump away from worst position (20%)
                
                global_pull = 0.3 * (global_best - pos)
                random_jump = 0.5 * np.random.normal(0, 0.5*panic_level, 2)
                
                # Find worst position among agents
                worst_idx = np.argmax([rastrigin(p) for p in positions])
                if worst_idx != i:
                    anti_pull = -0.2 * (positions[worst_idx] - pos)
                else:
                    anti_pull = np.zeros(2)
                
                proposal = pos + global_pull + random_jump + anti_pull
                proposal = np.clip(proposal, -5.12, 5.12)
                
                # Metropolis acceptance with panic temperature
                delta = rastrigin(proposal) - rastrigin(pos)
                temperature = 2.0 * panic_level
                
                if delta < 0 or np.random.random() < np.exp(-delta/temperature):
                    positions[i] = proposal
                
                # State transition: If found good spot, become geometric
                if rastrigin(positions[i]) < 3.0:
                    # 30% chance to switch to geometric state
                    if np.random.random() < 0.3:
                        states[i] = 0
            
            # === STATE 2: GRAVITATIONAL (MEMORY/ANCHOR) ===
            elif state == 2:
                # Slow, conservative movement toward best
                # But also has "gravitational lensing" - bends toward other agents
                
                # Primary pull to global best
                gravity = 0.08 * (global_best - pos)
                
                # Gravitational lensing: bend toward other good positions
                for j in range(3):
                    if i != j:
                        other_val = rastrigin(positions[j])
                        if other_val < rastrigin(pos) + 2.0:  # If other is somewhat better
                            delta = positions[j] - pos
                            dist = np.linalg.norm(delta) + 0.1
                            gravity += 0.03 * delta / dist
                
                # Tiny quantum fluctuation
                quantum = np.random.normal(0, 0.02, 2)
                
                proposal = pos + gravity + quantum
                proposal = np.clip(proposal, -5.12, 5.12)
                
                # Accept with small tolerance
                if rastrigin(proposal) < rastrigin(pos) + 0.2:
                    positions[i] = proposal
                
                # State transition: If not improving, become entropic
                if t > 30 and rastrigin(positions[i]) > global_best_val + 2.0:
                    if np.random.random() < 0.15:
                        states[i] = 1
        
        # === EMERGENT SYNERGY: QUANTUM ENTANGLEMENT ===
        # When two agents find similar good positions, they "entangle"
        # and share information instantaneously
        
        if t % 20 == 0:  # Check every 20 iterations
            # Find the best agent
            vals = [rastrigin(p) for p in positions]
            best_idx = np.argmin(vals)
            best_val = vals[best_idx]
            
            # If best agent found something really good
            if best_val < 2.0:
                # Teleport other agents NEAR (not to) the best
                for i in range(3):
                    if i != best_idx:
                        # Not exact copy - add controlled noise
                        positions[i] = positions[best_idx] + np.random.normal(0, 0.3, 2)
                        positions[i] = np.clip(positions[i], -5.12, 5.12)
                        
                        # Switch to geometric state to explore this new region
                        states[i] = 0
    
    return global_best_val < 1.0

def run_quantum_trinity():
    """
    QUANTUM TRINITY: Embracing superposition and collapse.
    
    The ultimate insight: The "ghost" exists in SUPERPOSITION of states.
    Each agent is ALL THREE STATES at once until "measured" (evaluated).
    
    This implements a quantum-inspired version where agents have
    probability amplitudes for each state, and collapse on measurement.
    """
    phi = 1.61803398875
    
    # Initialize 3 quantum agents
    # Each has a state vector: [amplitude_geom, amplitude_entropic, amplitude_grav]
    # These sum to 1 - they're in superposition
    amplitudes = [
        [0.4, 0.3, 0.3],  # Agent 1: leans geometric
        [0.3, 0.4, 0.3],  # Agent 2: leans entropic  
        [0.3, 0.3, 0.4]   # Agent 3: leans gravitational
    ]
    
    positions = [np.random.uniform(-5.12, 5.12, 2) for _ in range(3)]
    
    global_best = positions[0].copy()
    global_best_val = rastrigin(positions[0])
    
    # Quantum phase for interference
    phases = [0.0, 2*np.pi/3, 4*np.pi/3]
    
    for t in range(200):
        # Update global best
        for i, pos in enumerate(positions):
            val = rastrigin(pos)
            if val < global_best_val:
                global_best_val = val
                global_best = pos.copy()
        
        # Quantum evolution: amplitudes oscillate
        for i in range(3):
            # Rabi oscillation between states
            time_factor = np.sin(phi * t * 0.1 + phases[i])
            
            # Amplitude flows from worse-performing states to better ones
            # But preserves quantum uncertainty
            
            # Geometric amplitude gets boost if making progress
            if t > 0:
                current_val = rastrigin(positions[i])
                # Simple progress detection
                if current_val < 5.0:
                    amplitudes[i][0] *= (1.0 + 0.05 * time_factor)
                else:
                    amplitudes[i][1] *= (1.0 + 0.05 * time_factor)  # Boost entropic
            
            # Renormalize
            total = sum(amplitudes[i])
            if total > 0:
                amplitudes[i] = [a/total for a in amplitudes[i]]
        
        # === QUANTUM COLLAPSE AND BEHAVIOR ===
        for i in range(3):
            pos = positions[i]
            amp = amplitudes[i]
            
            # COLLAPSE: Choose behavior based on probability amplitudes
            # But with quantum interference - all states contribute
            
            # All three states contribute to the move
            geometric_component = np.zeros(2)
            entropic_component = np.zeros(2)
            gravitational_component = np.zeros(2)
            
            # 1. Geometric component (φ-harmonic)
            if amp[0] > 0.1:  # Only if amplitude significant
                angle = 2 * np.pi * phi * t + phases[i]
                step = 1.0 * amp[0] * np.exp(-t/50)
                geometric_component = step * np.array([np.sin(angle), np.cos(angle)])
            
            # 2. Entropic component (noise)
            if amp[1] > 0.1:
                noise_scale = 0.4 * amp[1] * (1 + np.sin(phi * t))
                entropic_component = np.random.normal(0, noise_scale, 2)
            
            # 3. Gravitational component (attraction)
            if amp[2] > 0.1:
                # Attraction to global best, weighted by amplitude
                gravity = 0.1 * amp[2] * (global_best - pos)
                
                # Also attraction to other agents' quantum states
                for j in range(3):
                    if i != j:
                        # Attraction proportional to other agent's geometric amplitude
                        # (Quantum entanglement: attracted to coherent states)
                        if amplitudes[j][0] > 0.5:  # Other agent is mostly geometric
                            delta = positions[j] - pos
                            dist = np.linalg.norm(delta) + 0.1
                            gravity += 0.05 * amp[2] * amplitudes[j][0] * delta / dist
                
                gravitational_component = gravity
            
            # Combine with quantum interference
            # The phases create constructive/destructive interference
            interference = np.sin(phases[i]) * 0.1
            
            proposal = (pos 
                       + geometric_component 
                       + entropic_component 
                       + gravitational_component
                       + interference * np.random.randn(2))
            
            # Quantum tunneling boundary condition
            # With small probability, tunnel through boundaries
            for dim in range(2):
                if proposal[dim] < -5.12:
                    if np.random.random() < amp[1] * 0.1:  # Entropic state can tunnel
                        # Tunnel to opposite side
                        proposal[dim] = 5.12 - (proposal[dim] + 5.12)
                    else:
                        proposal[dim] = -5.12 + abs(proposal[dim] + 5.12) * 0.8
                elif proposal[dim] > 5.12:
                    if np.random.random() < amp[1] * 0.1:
                        proposal[dim] = -5.12 + (proposal[dim] - 5.12)
                    else:
                        proposal[dim] = 5.12 - abs(proposal[dim] - 5.12) * 0.8
            
            # Quantum measurement: collapse affects acceptance
            current_val = rastrigin(pos)
            proposal_val = rastrigin(proposal)
            
            # The probability of acceptance depends on the state amplitudes
            # Geometric state wants improvement, entropic state is adventurous
            accept_prob = (amp[0] * (1.0 if proposal_val < current_val else 0.1) +
                          amp[1] * 0.5 +  # Entropic accepts 50% of moves
                          amp[2] * (0.8 if proposal_val < current_val + 0.5 else 0.2))
            
            if np.random.random() < accept_prob:
                positions[i] = proposal
        
        # === QUANTUM ENTANGMENT UPDATE ===
        # Agents that find good solutions become more "coherent" (geometric)
        # Agents that are stuck become more "decoherent" (entropic)
        for i in range(3):
            val = rastrigin(positions[i])
            
            if val < 2.0:  # Found good solution
                # Increase geometric amplitude, decrease entropic
                amplitudes[i][0] = min(1.0, amplitudes[i][0] * 1.1)
                amplitudes[i][1] = max(0.05, amplitudes[i][1] * 0.9)
            elif val > 10.0:  # Stuck in bad region
                # Increase entropic amplitude
                amplitudes[i][1] = min(1.0, amplitudes[i][1] * 1.2)
                amplitudes[i][0] = max(0.05, amplitudes[i][0] * 0.8)
            
            # Renormalize
            total = sum(amplitudes[i])
            if total > 0:
                amplitudes[i] = [a/total for a in amplitudes[i]]
    
    return global_best_val < 1.0

def run_adaptive_ensemble():
    """
    THE ULTIMATE: Adaptive ensemble of ALL strategies.
    
    Runs multiple solvers in parallel, then uses a meta-learner to
    combine their predictions and allocate resources.
    """
    # We'll run 5 different strategies
    n_strategies = 5
    positions = [np.random.uniform(-5.12, 5.12, 2) for _ in range(n_strategies)]
    performances = [0.0] * n_strategies  # Track success
    
    best_pos = positions[0].copy()
    best_val = rastrigin(positions[0])
    
    for t in range(300):  # More iterations for ensemble
        # Update best
        for i, pos in enumerate(positions):
            val = rastrigin(pos)
            if val < best_val:
                best_val = val
                best_pos = pos.copy()
            # Update performance (exponential moving average)
            performances[i] = 0.9 * performances[i] + 0.1 * (1.0 / (1.0 + val))
        
        # Normalize performances for resource allocation
        perf_sum = sum(performances)
        if perf_sum > 0:
            allocations = [p/perf_sum for p in performances]
        else:
            allocations = [1.0/n_strategies] * n_strategies
        
        # === STRATEGY 0: Pure φ-harmonic ===
        if allocations[0] > 0.1:  # Only if allocated resources
            step = 2.0 * np.exp(-t/60)
            angle = 2 * np.pi * 1.618 * t
            pos0 = positions[0] + step * np.array([np.sin(angle), np.cos(angle)])
            pos0 = np.clip(pos0, -5.12, 5.12)
            if rastrigin(pos0) < rastrigin(positions[0]):
                positions[0] = pos0
        
        # === STRATEGY 1: Gradient-aware ===
        if allocations[1] > 0.1:
            # Simple gradient approximation
            eps = 0.01
            grad = np.zeros(2)
            for dim in range(2):
                delta = np.zeros(2)
                delta[dim] = eps
                grad[dim] = (rastrigin(positions[1] + delta) - 
                            rastrigin(positions[1] - delta)) / (2*eps)
            
            # Move against gradient with noise
            pos1 = positions[1] - 0.1 * grad + np.random.normal(0, 0.05, 2)
            pos1 = np.clip(pos1, -5.12, 5.12)
            if rastrigin(pos1) < rastrigin(positions[1]):
                positions[1] = pos1
        
        # === STRATEGY 2: Entropic Trinity (our best) ===
        if allocations[2] > 0.1:
            # This agent implements the full asymmetric trinity internally
            # But as a single point in the ensemble
            pos = positions[2]
            
            # Simulate mini-trinity around this point
            for _ in range(5):  # 5 internal steps
                # Geometric component
                geom = np.random.normal(0, 0.1 * allocations[2], 2)
                # Entropic component
                ent = np.random.normal(0, 0.3, 2)
                # Gravitational component (to best)
                grav = 0.2 * (best_pos - pos)
                
                trial = pos + geom + ent + grav
                trial = np.clip(trial, -5.12, 5.12)
                
                if rastrigin(trial) < rastrigin(pos):
                    pos = trial
            
            positions[2] = pos
        
        # === STRATEGY 3: Pattern search ===
        if allocations[3] > 0.1:
            # Look in fibonacci spiral pattern
            phi = 1.618
            for k in range(5):
                angle = 2 * np.pi * phi * k
                radius = 0.2 * allocations[3] * np.exp(-k/3)
                trial = positions[3] + radius * np.array([np.sin(angle), np.cos(angle)])
                trial = np.clip(trial, -5.12, 5.12)
                
                if rastrigin(trial) < rastrigin(positions[3]):
                    positions[3] = trial
                    break
        
        # === STRATEGY 4: Teleportation ===
        if allocations[4] > 0.1:
            # Occasionally teleport to best position + noise
            if np.random.random() < 0.1 * allocations[4]:
                positions[4] = best_pos + np.random.normal(0, 0.5, 2)
                positions[4] = np.clip(positions[4], -5.12, 5.12)
            else:
                # Otherwise do simple local search
                trial = positions[4] + np.random.normal(0, 0.1, 2)
                trial = np.clip(trial, -5.12, 5.12)
                if rastrigin(trial) < rastrigin(positions[4]):
                    positions[4] = trial
        
        # === ENSEMBLE SYNERGY: Share discoveries ===
        if t % 30 == 0:
            # Find the best performing strategy
            best_strat = np.argmax(performances)
            
            # Other strategies occasionally jump toward best strategy's position
            for i in range(n_strategies):
                if i != best_strat and np.random.random() < 0.3:
                    # Jump partway toward best strategy
                    positions[i] = 0.7 * positions[i] + 0.3 * positions[best_strat]
                    positions[i] = np.clip(positions[i], -5.12, 5.12)
    
    return best_val < 1.0

# === MASSIVE TEST: PUSHING THE LIMITS ===
print("EXPLORING THE FRONTIER: Can we break 80%?")
print("=" * 70)

trials = 500  # Reduced for speed, but still significant
strategies = [
    ("Dynamic Trinity", run_dynamic_trinity),
    ("Quantum Trinity", run_quantum_trinity),
    ("Adaptive Ensemble", run_adaptive_ensemble)
]

print(f"\nRunning {trials} trials per strategy...")
results = {}

for name, func in strategies:
    print(f"\nTesting {name}...")
    wins = 0
    for i in range(trials):
        if i % 100 == 0:
            print(f"  Progress: {i}/{trials}")
        wins += func()
    success_rate = wins / trials * 100
    results[name] = success_rate
    print(f"  {name}: {success_rate:.2f}%")

print("\n" + "=" * 70)
print("FRONTIER RESULTS:")
print("=" * 70)
for name, rate in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:20s}: {rate:6.2f}%")
print("=" * 70)

# Compare with our baseline
print("\nIMPROVEMENT OVER BASELINE (0.9%):")
for name, rate in results.items():
    improvement = (rate / 0.9 - 1) * 100
    print(f"{name:20s}: {improvement:6.0f}x improvement")

# Theoretical maximum estimate
if max(results.values()) > 50:
    print("\n🎯 BREAKTHROUGH: We're now in uncharted territory!")
    print("   The Trinity concept scales beyond expectations.")
    
    if max(results.values()) > 70:
        print("   ⚡ 70%+ is now within reach!")
        print("   The next frontier: 90%+ with hybrid approaches.")
