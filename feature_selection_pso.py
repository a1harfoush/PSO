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
        return 1 / (1 + np.exp(-x))
    
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
