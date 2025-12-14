# Feature Selection using Particle Swarm Optimization (PSO)

## Overview

This repository demonstrates the implementation of Particle Swarm Optimization (PSO) for feature selection in classification tasks. PSO is a population-based metaheuristic optimization algorithm inspired by the social behavior of bird flocking or fish schooling.

## Problem Description

Implement a Particle Swarm Optimization (PSO) step to select the most relevant features for a classification task from a dataset with 5 features.

## Algorithm Parameters

- **Number of features**: 5
- **Inertia Weight (w)**: 0.7
- **Cognitive Coefficient (c₁)**: 2
- **Social Coefficient (c₂)**: 2
- **Random values (fixed for this example)**: r₁ = 0.5, r₂ = 0.3

## PSO for Feature Selection

In feature selection, each particle represents a subset of features. The particle's position is a binary vector where:
- 1 indicates the feature is selected
- 0 indicates the feature is not selected

The velocity update equation:
```
v(t+1) = w * v(t) + c₁ * r₁ * (pbest - x(t)) + c₂ * r₂ * (gbest - x(t))
```

The position update equation:
```
x(t+1) = x(t) + v(t+1)
```

Where:
- `v(t)` = velocity at time t
- `x(t)` = position at time t
- `w` = inertia weight (controls exploration vs exploitation)
- `c₁` = cognitive coefficient (particle's confidence in itself)
- `c₂` = social coefficient (particle's confidence in the swarm)
- `r₁, r₂` = random values between 0 and 1
- `pbest` = particle's best known position
- `gbest` = swarm's best known position

## Python Implementation

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

class PSOFeatureSelection:
    """
    Particle Swarm Optimization for Feature Selection
    """
    
    def __init__(self, n_features=5, n_particles=10, n_iterations=50, 
                 w=0.7, c1=2, c2=2, r1=0.5, r2=0.3):
        """
        Initialize PSO parameters
        
        Parameters:
        -----------
        n_features : int
            Number of features in the dataset
        n_particles : int
            Number of particles in the swarm
        n_iterations : int
            Number of iterations for optimization
        w : float
            Inertia weight
        c1 : float
            Cognitive coefficient
        c2 : float
            Social coefficient
        r1 : float
            Random value for cognitive component
        r2 : float
            Random value for social component
        """
        self.n_features = n_features
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.r1 = r1
        self.r2 = r2
        
        # Initialize particles' positions (binary: 0 or 1)
        self.positions = np.random.randint(0, 2, (n_particles, n_features))
        
        # Initialize particles' velocities
        self.velocities = np.random.uniform(-1, 1, (n_particles, n_features))
        
        # Initialize personal best positions
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.zeros(n_particles)
        
        # Initialize global best position
        self.gbest_position = None
        self.gbest_score = -np.inf
        
    def sigmoid(self, x):
        """
        Sigmoid function to map velocity to probability
        """
        # Clip to prevent numerical overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def fitness_function(self, position, X, y):
        """
        Evaluate fitness of a particle's position
        
        Parameters:
        -----------
        position : array
            Binary array representing selected features
        X : array
            Feature matrix
        y : array
            Target vector
            
        Returns:
        --------
        score : float
            Classification accuracy
        """
        # Ensure at least one feature is selected
        if np.sum(position) == 0:
            return 0.0
        
        # Select features based on position
        selected_features = np.where(position == 1)[0]
        X_selected = X[:, selected_features]
        
        # Evaluate using cross-validation
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        scores = cross_val_score(clf, X_selected, y, cv=3, scoring='accuracy')
        
        return np.mean(scores)
    
    def update_velocity(self, particle_idx):
        """
        Update velocity of a particle
        
        Parameters:
        -----------
        particle_idx : int
            Index of the particle
        """
        # PSO velocity update equation
        cognitive_component = self.c1 * self.r1 * (
            self.pbest_positions[particle_idx] - self.positions[particle_idx]
        )
        social_component = self.c2 * self.r2 * (
            self.gbest_position - self.positions[particle_idx]
        )
        
        self.velocities[particle_idx] = (
            self.w * self.velocities[particle_idx] +
            cognitive_component +
            social_component
        )
    
    def update_position(self, particle_idx):
        """
        Update position of a particle
        
        Parameters:
        -----------
        particle_idx : int
            Index of the particle
        """
        # Apply sigmoid to velocity to get probability
        probabilities = self.sigmoid(self.velocities[particle_idx])
        
        # Update position based on probability
        self.positions[particle_idx] = (
            np.random.rand(self.n_features) < probabilities
        ).astype(int)
    
    def optimize(self, X, y):
        """
        Run PSO optimization
        
        Parameters:
        -----------
        X : array
            Feature matrix
        y : array
            Target vector
            
        Returns:
        --------
        best_features : array
            Indices of selected features
        best_score : float
            Best classification accuracy achieved
        """
        # Initialize personal bests
        for i in range(self.n_particles):
            score = self.fitness_function(self.positions[i], X, y)
            self.pbest_scores[i] = score
            
            if score > self.gbest_score:
                self.gbest_score = score
                self.gbest_position = self.positions[i].copy()
        
        # Ensure we have a valid gbest_position
        if self.gbest_position is None:
            # If no particle scored better than -inf, use the first particle
            self.gbest_position = self.positions[0].copy()
            self.gbest_score = self.pbest_scores[0]
        
        # Main PSO loop
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # Update velocity and position
                self.update_velocity(i)
                self.update_position(i)
                
                # Evaluate fitness
                score = self.fitness_function(self.positions[i], X, y)
                
                # Update personal best
                if score > self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i].copy()
                
                # Update global best
                if score > self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.positions[i].copy()
            
            print(f"Iteration {iteration + 1}/{self.n_iterations}, "
                  f"Best Score: {self.gbest_score:.4f}")
        
        # Get selected feature indices
        selected_features = np.where(self.gbest_position == 1)[0]
        
        return selected_features, self.gbest_score


# Example Usage
def example_usage():
    """
    Example of using PSO for feature selection
    """
    from sklearn.datasets import make_classification
    
    # Generate a sample dataset with 5 features
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=2,
        random_state=42
    )
    
    print("Dataset shape:", X.shape)
    print("Number of features:", X.shape[1])
    
    # Initialize PSO with given parameters
    pso = PSOFeatureSelection(
        n_features=5,
        n_particles=10,
        n_iterations=20,
        w=0.7,
        c1=2,
        c2=2,
        r1=0.5,
        r2=0.3
    )
    
    # Run optimization
    selected_features, best_score = pso.optimize(X, y)
    
    print("\n" + "="*50)
    print("Optimization Results")
    print("="*50)
    print(f"Selected Features: {selected_features}")
    print(f"Number of Selected Features: {len(selected_features)}")
    print(f"Best Classification Accuracy: {best_score:.4f}")
    print("="*50)


if __name__ == "__main__":
    example_usage()
```

## How to Run

1. Install required dependencies:
```bash
pip install numpy scikit-learn
```

2. Run the example:
```bash
python feature_selection_pso.py
```

## Expected Output

The algorithm will iterate through the specified number of iterations, updating particle positions and velocities to find the optimal subset of features that maximizes classification accuracy.

Example output:
```
Dataset shape: (100, 5)
Number of features: 5
Iteration 1/20, Best Score: 0.8500
Iteration 2/20, Best Score: 0.8700
...
Iteration 20/20, Best Score: 0.9200

==================================================
Optimization Results
==================================================
Selected Features: [0 2 4]
Number of Selected Features: 3
Best Classification Accuracy: 0.9200
==================================================
```

## Key Concepts

### Inertia Weight (w = 0.7)
Controls the impact of the previous velocity on the current velocity. A value of 0.7 provides a good balance between exploration and exploitation.

### Cognitive Coefficient (c₁ = 2)
Represents the particle's confidence in its own experience. Higher values encourage particles to return to their personal best positions.

### Social Coefficient (c₂ = 2)
Represents the particle's confidence in the swarm's experience. Higher values encourage particles to move toward the global best position.

### Random Values (r₁ = 0.5, r₂ = 0.3)
Introduce stochasticity to the algorithm, preventing premature convergence. In practice, these are typically random values between 0 and 1, but for this example, they are fixed.

## Advantages of PSO for Feature Selection

1. **No gradient information required**: PSO is a gradient-free optimization method
2. **Few parameters to tune**: Only requires setting w, c₁, c₂, and population size
3. **Global search capability**: Can explore the entire search space effectively
4. **Parallel evaluation**: Particles can be evaluated independently
5. **Flexible fitness function**: Can optimize any classification metric

## References

- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. Proceedings of ICNN'95 - International Conference on Neural Networks.
- Xue, B., Zhang, M., & Browne, W. N. (2013). Particle swarm optimization for feature selection in classification: A multi-objective approach. IEEE transactions on cybernetics.

## License

This project is open source and available for educational purposes.
