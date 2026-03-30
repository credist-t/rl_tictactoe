import random
from collections import deque, Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



class ChemicalEnv:
    """
    Вещества:
    SO2, O2, SO3, H2O, H2SO4, H2SO3

    Реакции:
    1) SO2 + O2  -> SO3
    2) SO3 + H2O -> H2SO4
    3) SO2 + H2O -> H2SO3   (побочная)

    Действия:
    0 = add SO2
    1 = add O2
    2 = add H2O
    3 = wait
    4 = finish
    """

    def __init__(self, max_steps=10):
        self.max_steps = max_steps

        # коэффициенты скоростей
        self.k1 = 0.6
        self.k2 = 0.7
        self.k3 = 0.8

        self.reset()

    def reset(self):
        self.SO2 = 0.0
        self.O2 = 0.0
        self.SO3 = 0.0
        self.H2O = 0.0
        self.H2SO4 = 0.0
        self.H2SO3 = 0.0
        self.step_count = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        # Будем считатать разумным количеством вещества
        # в эксперименте 5.0 ммоль 
        return np.array([
            self.SO2 / 5.0,
            self.O2 / 5.0,
            self.SO3 / 5.0,
            self.H2O / 5.0,
            self.H2SO4 / 5.0,
            self.H2SO3 / 5.0,
            self.step_count / self.max_steps
        ], dtype=np.float32)

    def _clamp_nonnegative(self):
        self.SO2 = max(0.0, self.SO2)
        self.O2 = max(0.0, self.O2)
        self.SO3 = max(0.0, self.SO3)
        self.H2O = max(0.0, self.H2O)
        self.H2SO4 = max(0.0, self.H2SO4)
        self.H2SO3 = max(0.0, self.H2SO3)

    def _apply_reactions(self):
        # 1) SO2 + O2 -> SO3
        r1 = self.k1 * min(self.SO2, self.O2)
        self.SO2 -= r1
        self.O2 -= r1
        self.SO3 += r1

        # 2) SO3 + H2O -> H2SO4
        r2 = self.k2 * min(self.SO3, self.H2O)
        self.SO3 -= r2
        self.H2O -= r2
        self.H2SO4 += r2

        # 3) SO2 + H2O -> H2SO3
        r3 = self.k3 * min(self.SO2, self.H2O)
        self.SO2 -= r3
        self.H2O -= r3
        self.H2SO3 += r3

        self._clamp_nonnegative()

    def step(self, action):
        if self.done:
            raise ValueError("Episode is already finished. Call reset().")

        self.step_count += 1
        reward = -0.1  # штраф за шаг

        if action == 0:
            self.SO2 += 1.0
        elif action == 1:
            self.O2 += 1.0
        elif action == 2:
            self.H2O += 1.0
        elif action == 3:
            pass
        elif action == 4:
            self.done = True
        else:
            raise ValueError(f"Unknown action: {action}")

        if not self.done:
            self._apply_reactions()

            # промежуточное награждение:
            # полезно чуть-чуть награждать за H2SO4 и штрафовать за H2SO3
            reward += 0.5 * self.H2SO4 - 0.3 * self.H2SO3

        if self.step_count >= self.max_steps:
            self.done = True

        if self.done:
            reward += 10.0 * self.H2SO4 - 6.0 * self.H2SO3

        return self._get_state(), reward, self.done, {}

    def render(self):
        print(
            f"step={self.step_count} | "
            f"SO2={self.SO2:.3f}, O2={self.O2:.3f}, SO3={self.SO3:.3f}, "
            f"H2O={self.H2O:.3f}, H2SO4={self.H2SO4:.3f}, H2SO3={self.H2SO3:.3f}"
        )

    def get_raw_state_dict(self):
        return {
            "SO2": self.SO2,
            "O2": self.O2,
            "SO3": self.SO3,
            "H2O": self.H2O,
            "H2SO4": self.H2SO4,
            "H2SO3": self.H2SO3,
            "step_count": self.step_count
        }


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)



@dataclass
class Config:
    input_dim: int = 7
    output_dim: int = 5

    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 10000

    episodes: int = 400
    max_steps: int = 10

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.992

    target_update_freq: int = 20
    seed: int = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moving_average(values, window=20):
    if len(values) < window:
        return np.array(values, dtype=np.float32)
    kernel = np.ones(window) / window
    ma = np.convolve(values, kernel, mode="valid")
    prefix = [np.mean(values[:i+1]) for i in range(window - 1)]
    return np.array(prefix + ma.tolist(), dtype=np.float32)



def select_action(state, policy_net, epsilon, output_dim, device):
    if random.random() < epsilon:
        return random.randrange(output_dim)

    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_t)
    return int(torch.argmax(q_values, dim=1).item())



def train_step(policy_net, target_net, optimizer, replay_buffer, config, device):
    if len(replay_buffer) < config.batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(config.batch_size)

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    current_q = policy_net(states_t).gather(1, actions_t)

    with torch.no_grad():
        max_next_q = target_net(next_states_t).max(dim=1, keepdim=True)[0]
        target_q = rewards_t + config.gamma * max_next_q * (1.0 - dones_t)

    loss = nn.MSELoss()(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())



def train_dqn():
    config = Config()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ChemicalEnv(max_steps=config.max_steps)

    policy_net = DQN(config.input_dim, config.output_dim).to(device)
    target_net = DQN(config.input_dim, config.output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config.lr)
    replay_buffer = ReplayBuffer(config.buffer_capacity)

    epsilon = config.epsilon_start

    history = {
        "episode_rewards": [],
        "episode_avg_losses": [],
        "epsilons": [],
        "action_counts": Counter(),
        "final_h2so4": [],
        "final_h2so3": [],
        "final_so2": [],
        "final_o2": [],
        "final_so3": [],
        "final_h2o": [],
    }

    for episode in range(1, config.episodes + 1):
        state = env.reset()
        total_reward = 0.0
        losses = []

        for _ in range(config.max_steps):
            action = select_action(
                state=state,
                policy_net=policy_net,
                epsilon=epsilon,
                output_dim=config.output_dim,
                device=device
            )

            history["action_counts"][action] += 1

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            loss = train_step(policy_net, target_net, optimizer, replay_buffer, config, device)
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

            if done:
                break

        history["episode_rewards"].append(total_reward)
        history["episode_avg_losses"].append(np.mean(losses) if losses else np.nan)
        history["epsilons"].append(epsilon)

        history["final_so2"].append(env.SO2)
        history["final_o2"].append(env.O2)
        history["final_so3"].append(env.SO3)
        history["final_h2o"].append(env.H2O)
        history["final_h2so4"].append(env.H2SO4)
        history["final_h2so3"].append(env.H2SO3)

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

        if episode % config.target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 20 == 0:
            avg_reward = np.mean(history["episode_rewards"][-20:])
            last_losses = [x for x in history["episode_avg_losses"][-20:] if not np.isnan(x)]
            avg_loss = np.mean(last_losses) if last_losses else 0.0
            print(
                f"Episode {episode:3d} | "
                f"avg_reward={avg_reward:8.3f} | "
                f"epsilon={epsilon:6.3f} | "
                f"avg_loss={avg_loss:8.4f}"
            )

    return policy_net, env, history, config



def evaluate_agent(policy_net, env, episodes=5):
    device = next(policy_net.parameters()).device
    action_names = ["add SO2", "add O2", "add H2O", "wait", "finish"]

    eval_rewards = []
    final_states = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0

        print(f"\n=== Evaluation episode {ep} ===")
        env.render()

        while True:
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state_t)
            action = int(torch.argmax(q_values, dim=1).item())

            print(f"Action: {action_names[action]}")
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

            env.render()

            if done:
                print(f"Total reward: {total_reward:.3f}")
                eval_rewards.append(total_reward)
                final_states.append(env.get_raw_state_dict())
                break

    return eval_rewards, final_states



def plot_training_summary(history, eval_rewards=None, eval_final_states=None):
    rewards = history["episode_rewards"]
    losses = history["episode_avg_losses"]
    epsilons = history["epsilons"]

    reward_ma = moving_average(rewards, window=20)

    plt.figure(figsize=(14, 10))

    # 1. Reward
    plt.subplot(2, 3, 1)
    plt.plot(rewards, label="episode reward")
    plt.plot(reward_ma, label="moving avg (20)")
    plt.title("Награда по эпизодам")
    plt.xlabel("Эпизод")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    # 2. Loss
    plt.subplot(2, 3, 2)
    plt.plot(losses)
    plt.title("Средний loss по эпизодам")
    plt.xlabel("Эпизод")
    plt.ylabel("Loss")
    plt.grid(True)

    # 3. Epsilon
    plt.subplot(2, 3, 3)
    plt.plot(epsilons)
    plt.title("Падение epsilon")
    plt.xlabel("Эпизод")
    plt.ylabel("Epsilon")
    plt.grid(True)

    # 4. Распределение действий
    plt.subplot(2, 3, 4)
    action_names = ["SO2", "O2", "H2O", "wait", "finish"]
    counts = [history["action_counts"][i] for i in range(5)]
    plt.bar(action_names, counts)
    plt.title("Частоты действий")
    plt.ylabel("Count")
    plt.grid(True, axis="y")

    # 5. Финальные продукты в обучении
    plt.subplot(2, 3, 5)
    plt.plot(history["final_h2so4"], label="H2SO4")
    plt.plot(history["final_h2so3"], label="H2SO3")
    plt.title("Финальные продукты по эпизодам")
    plt.xlabel("Эпизод")
    plt.ylabel("Количество")
    plt.legend()
    plt.grid(True)

    # 6. Оценка после обучения
    plt.subplot(2, 3, 6)
    if eval_final_states:
        names = ["SO2", "O2", "SO3", "H2O", "H2SO4", "H2SO3"]
        means = []
        for name in names:
            means.append(np.mean([st[name] for st in eval_final_states]))
        plt.bar(names, means)
        plt.title("Средние финальные концентрации\nна evaluation")
        plt.ylabel("Amount")
        plt.grid(True, axis="y")
    else:
        plt.text(0.5, 0.5, "Нет данных evaluation", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    if eval_rewards is not None:
        print("\n=== Краткая сводка ===")
        print(f"Средняя награда за последние 20 эпизодов: {np.mean(rewards[-20:]):.3f}")
        print(f"Лучшая награда за обучение: {np.max(rewards):.3f}")
        print(f"Средняя награда на evaluation: {np.mean(eval_rewards):.3f}")
        print(f"Средний финальный H2SO4 на обучении: {np.mean(history['final_h2so4'][-20:]):.3f}")
        print(f"Средний финальный H2SO3 на обучении: {np.mean(history['final_h2so3'][-20:]):.3f}")



if __name__ == "__main__":
    policy_net, env, history, config = train_dqn()
    eval_rewards, eval_final_states = evaluate_agent(policy_net, env, episodes=3)
    plot_training_summary(history, eval_rewards, eval_final_states)
