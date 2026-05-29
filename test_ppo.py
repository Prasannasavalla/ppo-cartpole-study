import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

# 1. The Actor-Critic Brain
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(envs.single_observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(envs.single_observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, envs.single_action_space.n),
        )

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

# 2. Main Training Loop
if __name__ == "__main__":
    print("Initializing PPO Engine...")
    
    # Configuration
    total_timesteps = 100000
    num_steps = 128  # Rollout length
    lr = 2.5e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
    agent = Agent(envs)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    
    # Storage for historical results to build our chart later
    history_steps = []
    history_rewards = []
    
    obs, info = envs.reset()
    global_step = 0
    episodic_reward = 0
    
    print("\n--- Training Started (Watching Live Score Progression) ---")
    
    while global_step < total_timesteps:
        # Storage arrays for PPO batch collection
        obs_batch = torch.zeros((num_steps, 1) + envs.single_observation_space.shape)
        actions_batch = torch.zeros((num_steps, 1))
        logprobs_batch = torch.zeros((num_steps, 1))
        rewards_batch = torch.zeros((num_steps, 1))
        dones_batch = torch.zeros((num_steps, 1))
        values_batch = torch.zeros((num_steps, 1))
        
        # Step A: Collect data from environment (Rollout)
        for step in range(num_steps):
            global_step += 1
            obs_batch[step] = torch.Tensor(obs)
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(torch.Tensor(obs))
                values_batch[step] = value
                
            actions_batch[step] = action
            logprobs_batch[step] = logprob
            
            obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards_batch[step] = torch.Tensor(reward)
            
            done = terminations or truncations
            dones_batch[step] = torch.Tensor([done])
            episodic_reward += reward[0]
            
            if done:
                print(f"Step: {global_step:<7} | AI Score: {int(episodic_reward)}")
                history_steps.append(global_step)
                history_rewards.append(episodic_reward)
                episodic_reward = 0
                obs, info = envs.reset()

        # Step B: Calculate Advantages using GAE (Generalized Advantage Estimation)
        with torch.no_grad():
            next_value = agent.critic(torch.Tensor(obs)).reshape(1, -1)
            advantages = torch.zeros_like(rewards_batch)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - dones_batch[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_batch[t]
                    nextvalues = values_batch[t + 1]
                delta = rewards_batch[t] + gamma * nextvalues * nextnonterminal - values_batch[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_batch

        # Flatten arrays for training
        b_obs = obs_batch.reshape(-1, envs.single_observation_space.shape[0])
        b_logprobs = logprobs_batch.reshape(-1)
        b_actions = actions_batch.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Step C: PPO Brain Update Logic (Policy & Value loss optimization)
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions)
        logratio = newlogprob - b_logprobs
        ratio = logratio.exp()

        # PPO Safety Belt (Clipping Function)
        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value Loss & Entropy
        v_loss = 0.5 * ((newvalue.view(-1) - b_returns) ** 2).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()

    print("\nTraining Complete! Generating scientific analysis chart...")
    
    # Step D: Save a custom performance chart directly to disk
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(history_steps, history_rewards, color='#00ced1', linewidth=2, label='Agent Performance')
        plt.title('PPO Learning Curve: CartPole-v1 Study', fontsize=14, fontweight='bold')
        plt.xlabel('Global Training Steps', fontsize=12)
        plt.ylabel('Episodic Return (Score)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axhline(y=500, color='r', linestyle=':', label='Perfect Score (500)')
        plt.legend()
        plt.savefig('ppo_learning_curve.png')
        print("SUCCESS: 'ppo_learning_curve.png' saved directly to your project directory!")
    except ImportError:
        print("Matplotlib package not found. Run 'pip install matplotlib' to generate the physical graph image automatically next time!")