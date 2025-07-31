
import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir

def analyze_fading_memory(states, lr):
    # Calculate the Euclidean norm of states at each time step
    state_norms = np.linalg.norm(states, axis=1) #gives a measure of the overall activity in the reservoir


    print(np.shape(state_norms))

    # Plot the decay of state norms
    plt.figure(figsize=(10, 6))
    plt.plot(state_norms)
    plt.title(f"Fading Memory for  a leak rate of {lr}" )
    plt.xlabel("Time steps")
    plt.ylabel("State norm")
    plt.yscale('log')  # Use log scale for better visualization
    plt.grid(True)
    plt.show()

    # Calculate decay rate
    decay_rate = np.polyfit(np.arange(len(state_norms)), np.log(state_norms), 1)[0]
    print(f"Approximate decay rate: {decay_rate}")


lr = 1e-5
reservoir = Reservoir(units=1000, lr=lr, sr=0.99)
impulse = np.zeros((700, 8))
impulse[0] = [1,1,1,1,1,1,1,1]  # Single impulse at t=0
states = reservoir.run(impulse)
print(np.shape(states))
analyze_fading_memory(states, lr)