# 🧠 SARSA-Based Reinforcement Learning Algorithms

This repository provides clear implementations of various **SARSA-based reinforcement learning algorithms** using Python and OpenAI Gym. It includes:

- SARSA(0)
- SARSA-Max (also known as Q-learning)
- Expected SARSA

All algorithms use an ε-greedy exploration policy and are designed for **discrete action-space environments** like `Taxi-v3`, `CliffWalking-v0`, and `FrozenLake-v1`.

---

## 📌 Algorithms Explained

### 1. 🔁 SARSA(0)
SARSA is an **on-policy TD control algorithm** that updates Q-values using the next action selected by the same policy.

**Update Rule:**

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma Q(s', a') - Q(s, a)\right]
\]

Where:
- `s, a` = current state and action  
- `s′, a′` = next state and action  
- `r` = reward  
- `α` = learning rate  
- `γ` = discount factor  

---

### 2. 📈 SARSA-Max (Q-Learning)
Q-Learning is an **off-policy** variant of SARSA. It uses the maximum future Q-value to update the current state-action pair.

**Update Rule:**

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]
\]

It assumes the agent will always act optimally in the future (greedy update).

---

### 3. 📊 Expected SARSA
Expected SARSA averages over all possible actions using their probabilities under the current ε-greedy policy. This leads to more **stable learning**.

**Update Rule:**

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \sum_{a'} \pi(a'|s') Q(s', a') - Q(s, a)\right]
\]

Where `π(a′|s′)` is the probability of action `a′` under the current ε-greedy policy.

---

## 🧪 Example Environments

These algorithms have been tested on:

- `Taxi-v3`
- `CliffWalking-v0`
- `FrozenLake-v1` (both deterministic and stochastic)

All environments are part of [OpenAI Gym](https://www.gymlibrary.dev/).

---

## 🚀 How to Use

### 1. 🔧 Installation

```bash
git clone https://github.com/your-username/sarsa-algorithms.git
cd sarsa-algorithms
pip install -r requirements.txt
