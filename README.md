### Introduction
Welcome to FrozenLakeAgent, a Python-based implementation of a Q-learning agent designed to navigate the popular Frozen Lake environment from the Gymnasium library. This project explores reinforcement learning techniques, specifically Q-learning, to train an agent capable of traversing a slippery, grid-based frozen lake without falling into holes. By adjusting parameters like learning rate (alpha), discount factor (gamma), and exploration rate (epsilon), the agent learns optimal strategies through trial and error.

### Purpose
The purpose of this project is to provide a practical implementation of reinforcement learning using the Q-learning algorithm, a fundamental concept in machine learning. The project serves as both a learning tool and a demonstration of how an agent can be trained in uncertain environments. It is particularly useful for those looking to deepen their understanding of reinforcement learning principles, hyperparameter tuning, and applying Gymnasium environments in practice.

The key goals of the project include:
- Training an agent to navigate Frozen Lake environments using Q-learning.
- Providing visualization tools to track the learning progress and agent performance.
- Demonstrating how Q-tables are used to store knowledge and improve decision-making over time.

### How to Run
##### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Gymnasium library (`pip install gymnasium`)
- Numpy (`pip install numpy`)
- Matplotlib (`pip install matplotlib`)
##### Running the Project
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Q-LakeAgen.git
cd Q-LakeAgen
```
2. Train the agent:

```bash
python main.py
```

3. (Optional) To visualize the agent's performance in the Frozen Lake environment with rendering enabled:

```bash
python main.py --episodes 10 --render
```
4. Evaluate the agent after training by loading the saved Q-table:

```bash
python main.py --episodes 10 --evaluate
```
### Notes:
- Training progress will be saved in result.pkl.
- A graph showing cumulative rewards will be saved as frozen_lake8x8.png.
