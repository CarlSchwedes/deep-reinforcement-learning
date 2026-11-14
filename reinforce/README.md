[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135683-dde5c6f0-7d13-11e8-90b1-8770df3e40cf.gif "Trained Agent"

# REINFORCE

### Instructions

Open `REINFORCE.ipynb` to see an implementation of REINFORCE (also known as Monte Carlo Policy Gradients) with OpenAI Gym's Cartpole environment.

Try to change the parameters in the notebook, to see if you can get the agent to train faster!

### Results

![Trained Agent][image1]


## Algorithm

REINFORCE is a **policy gradient reinforcement learning algorithm** that learns an optimal policy by directly optimizing the probability of selecting actions that maximize the expected cumulative reward.

Instead of estimating value functions such as state values or action values, REINFORCE updates the policy parameters using the rewards collected from complete episodes (classical Monte Carlo REINFORCE using complete terminal-state trajectories).

---

## Working

### 1. Collect Episodes

The agent interacts with the environment for a fixed number of steps or until an episode is complete, following the current policy.

This generates a trajectory consisting of states, actions, and rewards.

A trajectory can be written as

$$
\tau =
(s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_T).
$$

---

### 2. Calculate Returns

For each time step $t$, calculate the return $G_t$, which is the total discounted reward obtained from time step $t$ onwards.

Typically, this is the discounted sum of rewards:

$$
G_t = \sum_{k=t}^{T} \gamma^{k-t} R_k
$$

where:

- $\gamma$ is the **discount factor**,
- $T$ is the final time step of the episode,
- $R_k$ is the reward received at time step $k$.

Equivalently,

$$
G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots + \gamma^{T-t}R_T.
$$

The discount factor usually satisfies

$$
0 \leq \gamma \leq 1.
$$

---

### 3. Policy Gradient Update

The policy parameters $\theta$ are updated according to

$$
\theta_{t+1} = \theta_t + \alpha
\nabla_\theta
\log \pi_\theta(a_t \mid s_t)
G_t.
$$

### Where

- $\alpha$ is the **learning rate**.
- $\pi_\theta(a_t \mid s_t)$ is the probability of taking action $a_t$ in state $s_t$ according to the policy.
- $G_t$ is the return or cumulative discounted reward obtained from time step $t$ onwards.
- The gradient

$$
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

represents how the policy parameters influence the log-probability of selecting action $a_t$ in state $s_t$.

The return $G_t$ determines whether the probability of the selected action should effectively be increased or decreased.

---

### 4. Repeat

This process is repeated for several episodes, iteratively updating the policy in the direction of higher expected rewards.

---

## Classical REINFORCE Uses Complete Episodes

In the original **Monte Carlo REINFORCE algorithm**, the agent normally collects a **complete trajectory until a terminal state**:

$$
\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_T)
$$

Only once the episode ends do we know the actual Monte Carlo returns.

For each time step, however, REINFORCE does **not** necessarily use the reward of the entire episode.

Instead, it calculates the **reward-to-go** from that particular time step:

$$
G_t = \sum_{k=t}^{T} \gamma^{k-t} R_k
$$

So:

$$
G_0 = R_0 + \gamma R_1 + \gamma^2 R_2 + \cdots
$$

$$
G_1 = R_1 + \gamma R_2 + \gamma^2 R_3 + \cdots
$$

$$
G_2 = R_2 + \gamma R_3 + \cdots
$$

and so forth.

The total update is approximately

$$
\boxed{
\nabla_\theta J(\theta)
\approx
\sum_{t=0}^{T}
G_t
\nabla_\theta
\log \pi_\theta(a_t \mid s_t)
}
$$

So there's an important nuance:

> **REINFORCE waits for the complete episode, but it learns separately from every state-action decision within that episode.**

---

## Why Does It Wait Until the End?

This comes from REINFORCE being a **Monte Carlo method**.

At time $t=2$, suppose you're here:

```text
s0 -> s1 -> s2 -> ?
            ^
           now
```

You know rewards that have already happened, but you don't yet know:

$$
R_3, R_4, \ldots, R_T.
$$

Therefore, you cannot yet calculate the actual return

$$
G_2 = R_2 + \gamma R_3 + \gamma^2 R_4 + \cdots.
$$

REINFORCE essentially says:

> "I'll wait and see what actually happens."

After the episode finishes:

```text
s0 -> s1 -> s2 -> s3 -> s4 -> TERMINAL
            ↑
       now known
```

the entire sequence of rewards is available, so $G_2$ can be calculated exactly from the observed trajectory.

This is why REINFORCE is called a **Monte Carlo policy-gradient method**.

---

## But This Creates a Major Limitation

Suppose you're training a robot and an episode lasts **30 minutes**.

At minute 2, it performs an excellent action.

Basic REINFORCE **doesn't update its policy immediately**.

It continues:

$$
t=2\rightarrow 3\rightarrow 4\rightarrow\cdots\rightarrow T
$$

Only when the episode finishes can it calculate the actual Monte Carlo returns and perform the policy update.

This can be inefficient.


## This Is Where Actor-Critic Becomes Interesting

Instead of waiting until $T$, we can **estimate the unknown future return**.

At time $t$, write:

$$
G_t
=
R_{t+1}
+
\gamma
\underbrace{
\left(
R_{t+2}
+
\gamma R_{t+3}
+
\cdots
\right)
}_{\text{future not observed yet}}
$$

REINFORCE waits to observe that future.

An Actor-Critic algorithm instead has a **critic** estimating

$$
V_\phi(s_{t+1})
\approx
\mathbb{E}
\left[
G_{t+1}
\mid
s_{t+1}
\right].
$$

Then, after just one transition

$$
(s_t, a_t, r_{t+1}, s_{t+1}),
$$

it can form the **TD error**:

$$
\boxed{
\delta_t
=
r_{t+1}
+
\gamma V_\phi(s_{t+1})
-
V_\phi(s_t)
}
$$

and use it almost immediately:

$$
\boxed{
\theta
\leftarrow
\theta
+
\alpha
\delta_t
\nabla_\theta
\log \pi_\theta(a_t \mid s_t)
}
$$

Now we no longer necessarily need to wait for the terminal state.

---

## This Gives Us a Useful Spectrum

We can think of these methods as different ways of answering the same question:

> How much of the future should we actually observe before estimating the remaining return?

### REINFORCE: Wait for Everything

```text
s0 -> s1 -> s2 -> s3 -> s4 -> TERMINAL
|_____________________________________|
              observed return
```

The complete return is observed:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1}R_T.
$$

No learned estimate of the remaining future is necessary.

---

### One-Step Actor-Critic: Wait One Transition

```text
st -> st+1 -> ?
|_________|
 observed
```

The immediate reward is observed, while the remaining future return is estimated:

$$
G_t\approx R_{t+1} + \gamma V(s_{t+1}).
$$

The critic therefore replaces the unknown future rewards with its current estimate of the value of the next state:

$$
V(s_{t+1})\approx\mathbb{E}\left[G_{t+1}\mids_{t+1}\right].
$$

This gives the TD error:

$$
\delta_t = R_{t+1}
+
\gamma V(s_{t+1})
-
V(s_t).
$$

---

### n-Step Actor-Critic: Temporary Sequences

There is also an intermediate solution between observing one transition and observing the complete episode.

Collect a temporary sequence of $n$ transitions:

```text
st -> st+1 -> st+2 -> ... -> st+n -> ?
|_____________________________|
         observed n steps
```

Then calculate the **n-step return**:

$$
\boxed{
G_t^{(n)} = R_{t+1} + \gamma R_{t+2}
+
\cdots
+
\gamma^{n-1}R_{t+n}
+
\gamma^n V(s_{t+n})
}
$$

The first $n$ rewards are actually observed:

$$
R_{t+1} + \gamma R_{t+2}
+
\cdots
+
\gamma^{n-1}R_{t+n},
$$

while the remaining future is estimated by

$$
\gamma^n V(s_{t+n}).
$$

For example, with $n=3$:

$$
G_t^{(3)} = R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2 R_{t+3}
+
\gamma^3 V(s_{t+3}).
$$

This very literally **learns from temporary trajectory sequences**.

---

# The Deeper Distinction: Monte Carlo vs. Bootstrapping

This distinction is more fundamental than simply REINFORCE vs. Actor-Critic.

## Monte Carlo

Monte Carlo methods wait until the future is actually observed:

$$
G_t = R_{t+1}
+
\gamma R_{t+2}
+
\cdots
+
\gamma^{T-t-1}R_T.
$$

The complete return comes from actual observations.

**No bootstrapping is required.**

Conceptually:

```text
Present                                Terminal
   |                                       |
   v                                       v

   st -> st+1 -> st+2 -> ... -> sT
         |        |              |
       Rt+1     Rt+2             RT
   |_______________________________________|
              actually observed
```

---

## Temporal Difference / Actor-Critic

Temporal-Difference methods observe some rewards and replace the unknown remainder with an estimate.

For an $n$-step return:

$$
G_t^{(n)} = \underbrace{
R_{t+1}
+
\gamma R_{t+2}
+
\cdots
+
\gamma^{n-1}R_{t+n}
}_{\text{actually observed}}
+
\underbrace{
\gamma^n V(s_{t+n})
}_{\text{estimated future}}.
$$

The estimated part

$$
V(s_{t+n})
$$

is itself a learned value estimate.

Using one learned estimate to construct a target for another update is called **bootstrapping**.

Conceptually:

```text
Present               n observed steps
   |                         |
   v                         v

   st -> st+1 -> ... -> st+n -> future
          |              |
        Rt+1           Rt+n
   |_____________________|
      actually observed

                         |---------->
                         V(st+n)
                      estimated future
```

---

## Bias-Variance Trade-Off

Choosing how many real rewards to observe before bootstrapping introduces an important trade-off:

| Method | Wait | Future Return | Variance | Bias |
|---|---:|---|---:|---:|
| REINFORCE / Monte Carlo | Full episode | Fully observed | High | Low |
| n-step Actor-Critic | $n$ steps | Part observed, part estimated | Medium | Medium |
| 1-step Actor-Critic | 1 step | Mostly estimated | Lower | Higher |

### Monte Carlo extreme

At one extreme:

$$
n = T-t
$$

which gives the complete Monte Carlo return:

$$
G_t = R_{t+1}
+
\gamma R_{t+2}
+
\cdots
+
\gamma^{T-t-1}R_T.
$$

There is no value estimate at the end because the terminal state has no remaining future return.

### One-step TD extreme

At the other extreme:

$$
n=1,
$$

which gives:

$$
G_t^{(1)} = R_{t+1}
+
\gamma V(s_{t+1}).
$$

Only one reward is actually observed before bootstrapping.

### Intermediate n-step return

Between these extremes:

$$
1<n<T-t,
$$

giving:

$$
G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n V(s_{t+n}).
$$

Therefore, the different approaches can be viewed as a spectrum:

```text
More bootstrapping                                 More sampling
Lower variance                                         Higher variance
Higher bias                                               Lower bias

1-step TD -------- n-step TD -------------------- Monte Carlo
    |                  |                              |
    v                  v                              v

 R + gamma V     R + ... + gamma^n V       Complete sampled return
```

This is one of the central ideas connecting **REINFORCE**, **Temporal-Difference learning**, and modern **Actor-Critic** methods.



---


# Implementation

In this example, we train a policy network to solve a basic environment such as **CartPole** from OpenAI Gym.

The aim is to use REINFORCE to directly optimize the policy without using value-function approximations.

Udacity links cart pole examples:
- [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/)
- [kvfrans-CartPole](https://kvfrans.com/simple-algoritms-for-solving-cartpole/)
- [kvfrans-github](https://github.com/kvfrans/openai-cartpole)

---

## Step 1: Set Up the Environment

The first step is to create the environment using OpenAI Gym.

For this example, we use the `CartPole-v1` environment, where the agent's task is to balance a pole on a cart.

```python
import gym
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers


env = gym.make("CartPole-v1")

obs_space = env.observation_space.shape[0]
act_space = env.action_space.n
```

---

## Step 2: Define Hyperparameters

In this step, we define hyperparameters for the algorithm such as the discount factor $\gamma$, learning rate, number of episodes, and batch size.

These hyperparameters control how the algorithm behaves during training.

```python
gamma = 0.99
learning_rate = 0.01
num_episodes = 1000
batch_size = 64
```

where:

- $\gamma = 0.99$ is the discount factor,
- $\alpha = 0.01$ is the learning rate,
- `num_episodes = 1000` determines the number of training episodes,
- `batch_size = 64` defines the nominal batch size.

> **Note:** In the code below, `batch_size` is defined but is not actually used. The policy update is performed once per complete episode.

---

## Step 3: Define the Policy Network (Actor)

We define the policy network as a simple neural network with two dense layers.

The input to the network is the state, and the output is a probability distribution over the available actions using a softmax output layer.

Conceptually, the network represents

$$
\pi_\theta(a \mid s).
$$

For a discrete action space with $n$ actions,

$$
\sum_{a=1}^{n}\pi_\theta(a \mid s)=1.
$$

```python
class PolicyNetwork(tf.keras.Model):
    def __init__(self, hidden_units=128):
        super(PolicyNetwork, self).__init__()

        self.dense1 = layers.Dense(
            hidden_units,
            activation="relu"
        )

        self.dense2 = layers.Dense(
            env.action_space.n,
            activation="softmax"
        )

    def call(self, state):
        x = self.dense1(state)
        return self.dense2(x)
```

---

## Step 4: Initialize the Policy & Optimizer

Here, we initialize the policy network and the Adam optimizer.

The optimizer is used to update the weights of the policy network during training.

```python
policy = PolicyNetwork()

optimizer = tf.keras.optimizers.Adam(learning_rate)
```

---

## Step 5: Compute Returns

In reinforcement learning, the return $G_t$ is the discounted sum of future rewards:

$$
G_t =
\sum_{k=t}^{T}
\gamma^{k-t}R_k.
$$

It can also be calculated recursively:

$$
G_t = R_t + \gamma G_{t+1}.
$$

The following function computes the return for every time step $t$ based on the rewards collected during an episode.

```python
def compute_returns(rewards, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)

    running_return = 0

    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns
```

---

## Step 6: Define Training Step

The training step computes the gradients of the policy network using the log-probabilities of the actions that were actually selected and the corresponding returns.

For a single time step, the REINFORCE objective is based on

$$
\log \pi_\theta(a_t \mid s_t) G_t.
$$

For an entire trajectory, the policy objective can be written as

$$
J(\theta) =
\sum_{t=0}^{T}
\log \pi_\theta(a_t \mid s_t)G_t.
$$

Because TensorFlow optimizers **minimize** a loss function, the implementation uses the negative objective:

$$
\mathcal{L}(\theta) =
-
\frac{1}{N}
\sum_{t=1}^{N}
G_t
\log \pi_\theta(a_t \mid s_t).
$$

```python
def train_step(states, actions, returns):
    with tf.GradientTape() as tape:

        # Calculate the probability of each action
        action_probs = policy(states)

        action_indices = np.array(
            actions,
            dtype=np.int32
        )

        # Gather the probabilities of the selected actions
        action_log_probs = tf.math.log(
            tf.reduce_sum(
                action_probs
                * tf.one_hot(
                    action_indices,
                    env.action_space.n
                ),
                axis=1
            )
        )

        # Negative policy-gradient objective
        loss = -tf.reduce_mean(
            action_log_probs * returns
        )

    grads = tape.gradient(
        loss,
        policy.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            grads,
            policy.trainable_variables
        )
    )
```

---

## Step 7: Training Loop

The training loop collects experiences from complete episodes.

For each episode:

1. Reset the environment.
2. Observe the current state.
3. Calculate the action probabilities.
4. Sample an action from the current policy.
5. Execute the action.
6. Store states, actions, and rewards.
7. Compute the returns after the episode ends.
8. Update the policy.

The action is sampled according to

$$
a_t \sim \pi_\theta(\cdot \mid s_t).
$$

```python
for episode in range(num_episodes):

    state, _ = env.reset()
    done = False

    states = []
    actions = []
    rewards = []

    while not done:

        state_input = np.array(
            state,
            dtype=np.float32
        ).reshape(1, -1)

        probs = policy(state_input).numpy()[0]

        action = np.random.choice(
            act_space,
            p=probs
        )

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        states.append(state_input[0])
        actions.append(action)
        rewards.append(reward)

        state = next_state

    # Compute discounted returns after the episode
    returns = compute_returns(
        rewards,
        gamma
    )

    # Normalize returns
    returns = (
        returns - np.mean(returns)
    ) / (
        np.std(returns) + 1e-9
    )

    states_batch = np.vstack(states)

    train_step(
        states_batch,
        actions,
        returns
    )

    if episode % 100 == 0:
        print(
            f"Episode {episode}/{num_episodes}"
        )
```

The normalization step

$$
\hat{G}_t =
\frac{
G_t-\mu_G
}{
\sigma_G+\epsilon
}
$$

uses the mean $\mu_G$ and standard deviation $\sigma_G$ of the returns from the episode.

In the implementation,

$$
\epsilon = 10^{-9}
$$

is added for numerical stability.

---

## Step 8: Testing the Trained Agent

After training, we evaluate the policy without updating its parameters.

Instead of sampling from the action distribution, the implementation selects the action with the highest predicted probability:

$$
a_t =
\underset{a}{\arg\max}
\;
\pi_\theta(a \mid s_t).
$$

```python
state, _ = env.reset()

done = False
total_reward = 0

while not done:

    state_input = np.array(
        state,
        dtype=np.float32
    ).reshape(1, -1)

    probs = policy(state_input).numpy()[0]

    action = np.argmax(probs)

    next_state, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated

    total_reward += reward
    state = next_state

print(
    f"Test Total Reward: {total_reward}"
)
```

---

## Output

```text
Episode 0/1000
Episode 100/1000
Episode 200/1000
Episode 300/1000
Episode 400/1000
Episode 500/1000
Episode 600/1000
Episode 700/1000
Episode 800/1000
Episode 900/1000
Test Total Reward: 49.0
```

---

# Variants of the REINFORCE Algorithm

Several modifications to the original REINFORCE algorithm have been proposed to address its high variance.

## REINFORCE with Baseline

A common improvement is to subtract a **baseline** $b_t$ from the return:

$$
G_t - b_t.
$$

The baseline is typically an estimate of the state-value function:

$$
b_t = V(s_t).
$$

Subtracting an action-independent baseline can reduce the variance of the policy-gradient estimator without changing its expectation.

The update rule becomes

$$
\theta_{t+1} = \theta_t
+
\alpha
\nabla_\theta
\log \pi_\theta(a_t \mid s_t)
\left(G_t-b_t\right).
$$

If the baseline is the state-value function,

$$
\theta_{t+1} = \theta_t
+
\alpha
\nabla_\theta
\log \pi_\theta(a_t \mid s_t)
\left(G_t-V(s_t)\right).
$$

The quantity

$$
G_t - V(s_t)
$$

can be interpreted as a Monte Carlo estimate of the **advantage** of the selected action.

---

## Actor-Critic

Actor-Critic methods use two components:

### Actor

The **actor** represents the policy

$$
\pi_\theta(a \mid s)
$$

and determines which action should be selected.

### Critic

The **critic** estimates a value function, for example

$$
V_\phi(s),
$$

where $\phi$ denotes the critic's parameters.

Instead of using the complete Monte Carlo return directly, the critic can provide an estimate of how good the selected action or current state is.

A common temporal-difference error is

$$
\delta_t = R_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t).
$$

This signal can then be used to update the actor:

$$
\theta\leftarrow\theta + \alpha\nabla_\theta\log \pi_\theta(a_t \mid s_t)\delta_t.
$$

This generally reduces variance and allows learning before an entire episode has finished.

---

# Applications

- **Robotics:** REINFORCE can be used to learn control policies for tasks such as manipulation, locomotion, and object interaction.

- **Game AI:** Policy-gradient methods can learn strategies by repeatedly interacting with a game environment and reinforcing actions associated with high returns.

- **Autonomous systems:** Policy-gradient methods can be investigated for sequential decision-making tasks in simulated autonomous-driving environments.

---

# Advantages

1. **Simple policy-gradient formulation:** REINFORCE provides one of the most direct implementations of the policy-gradient theorem.

2. **Direct policy optimization:** The algorithm directly optimizes a parameterized policy rather than deriving one indirectly from a learned action-value function.

3. **Supports stochastic policies:** Actions can be sampled directly from

   $$
   \pi_\theta(a \mid s).
   $$

4. **Applicable to episodic tasks:** REINFORCE naturally works with environments where complete trajectories and their cumulative rewards can be collected.

5. **Foundation for more advanced algorithms:** The central idea behind REINFORCE leads naturally to actor-critic and more advanced policy-gradient methods.

---

# Challenges

1. **High variance:** Monte Carlo policy-gradient estimates can exhibit substantial variance.

2. **Sample inefficiency:** REINFORCE may require many interactions with the environment before obtaining a good policy.

3. **Delayed updates:** In its basic form, the policy is updated only after a complete episode has been collected.

4. **Sparse rewards:** Learning can be particularly slow when useful rewards are rare.

5. **Sensitivity to hyperparameters:** Performance can depend strongly on the learning rate, discount factor, policy architecture, and return normalization.

6. **Training instability:** Since policy updates depend directly on sampled trajectories, the gradient can vary significantly between episodes.

---

# Related Topics

- Reinforcement Learning
- Monte Carlo Methods
- Policy Gradient Methods
- Policy Gradient Theorem
- Stochastic Policies
- REINFORCE with Baseline
- Advantage Functions
- Actor-Critic Methods
- Deep Q-Networks (DQN)