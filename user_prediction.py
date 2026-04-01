from typing import Dict, List
import numpy as np
from dataclasses import dataclass


ARMS = ["sport", "education", "entertainment", "technology"]


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class UserProfile:
    age: int
    gender: int
    is_student: bool
    works_in_it: bool
    sports_interest: float
    tech_interest: float
    education_interest: float
    entertainment_interest: float

    def to_feature_vector(self, is_evening: bool, is_weekend: bool) -> np.ndarray:
        age_scaled = (self.age - 35.0) / 15.0
        return np.array(
            [
                1.0,
                age_scaled,
                float(self.gender),
                float(self.is_student),
                float(self.works_in_it),
                float(self.sports_interest),
                float(self.tech_interest),
                float(self.education_interest),
                float(self.entertainment_interest),
                float(is_evening),
                float(is_weekend),
            ],
            dtype=float,
        )


class NewsEnvironment:
    def __init__(self, true_theta: Dict[str, np.ndarray], rng: np.random.Generator | None = None) -> None:
        self.true_theta = true_theta
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def get_reward(self, arm: str, x: np.ndarray) -> int:
        logit = float(self.true_theta[arm] @ x)
        p = sigmoid(logit)
        reward = int(self.rng.random() < p)
        return reward

    def click_probability(self, arm: str, x: np.ndarray) -> float:
        return sigmoid(float(self.true_theta[arm] @ x))


class LinUCBArm:
    def __init__(self, dim: int):
        self.dim = dim
        self.A = np.eye(dim, dtype=float)
        self.b = np.zeros(dim, dtype=float)

    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def ucb_score(self, x: np.ndarray, alpha: float) -> float:
        theta_loc = self.theta()
        exploit = float(theta_loc @ x)
        z = np.linalg.solve(self.A, x)
        explore = alpha * np.sqrt(float(x @ z))
        return exploit + explore

    def update(self, x: np.ndarray, reward: float) -> None:
        self.A += np.outer(x, x)
        self.b += reward * x


class LinUCBAgent:
    def __init__(self, arms: List[str], dim: int, alpha: float = 1.0):
        self.alpha = alpha
        self.arms: Dict[str, LinUCBArm] = {arm: LinUCBArm(dim) for arm in arms}

    def predict_scores(self, x: np.ndarray) -> Dict[str, float]:
        return {arm: model.ucb_score(x, self.alpha) for arm, model in self.arms.items()}

    def select_arm(self, x: np.ndarray) -> str:
        scores = self.predict_scores(x)
        return max(scores, key=scores.get)

    def update(self, arm: str, x: np.ndarray, reward: float) -> None:
        self.arms[arm].update(x, reward)


def sample_user(rng: np.random.Generator) -> UserProfile:
    age = int(rng.integers(18, 61))
    gender = int(rng.integers(0, 2))
    is_student = bool(rng.random() < 0.35)
    works_in_it = bool(rng.random() < 0.25)
    sports_interest = float(rng.uniform(0.0, 1.0))
    tech_interest = float(rng.uniform(0.0, 1.0))
    education_interest = float(rng.uniform(0.0, 1.0))
    entertainment_interest = float(rng.uniform(0.0, 1.0))

    return UserProfile(
        age=age,
        gender=gender,
        is_student=is_student,
        works_in_it=works_in_it,
        sports_interest=sports_interest,
        tech_interest=tech_interest,
        education_interest=education_interest,
        entertainment_interest=entertainment_interest,
    )


def build_hidden_theta(dim: int) -> Dict[str, np.ndarray]:
    thetas = {
        "technology": np.array([
            -0.2,
            -0.05,
            0.05,
            0.10,
            0.90,
            0.00,
            1.40,
            0.20,
            0.10,
            0.10,
            0.00,
        ], dtype=float),
        "sport": np.array([
            -0.1,
            -0.10,
            0.20,
            0.00,
            0.00,
            1.50,
            0.10,
            0.00,
            0.10,
            0.20,
            0.40,
        ], dtype=float),
        "entertainment": np.array([
            0.00,
            -0.15,
            0.00,
            0.10,
            0.00,
            0.10,
            0.00,
            0.00,
            1.50,
            0.50,
            0.30,
        ], dtype=float),
        "education": np.array([
            -0.15,
            -0.05,
            0.00,
            0.80,
            0.10,
            0.00,
            0.20,
            1.30,
            0.00,
            -0.10,
            -0.05,
        ], dtype=float),
    }

    for arm, theta in thetas.items():
        if theta.shape[0] != dim:
            raise ValueError(f"Arm {arm}: expected dim={dim}, got {theta.shape[0]}")
    return thetas


def main() -> None:
    rand_state = np.random.choice(range(1, 1000))
    rng = np.random.default_rng(rand_state)

    dim = 11
    alpha = 1.0
    n_steps = 50000

    true_theta = build_hidden_theta(dim)
    env = NewsEnvironment(true_theta=true_theta, rng=rng)
    agent = LinUCBAgent(arms=ARMS, dim=dim, alpha=alpha)

    total_reward = 0
    arm_counts = {arm: 0 for arm in ARMS}
    arm_rewards = {arm: 0 for arm in ARMS}

    for step in range(1, n_steps + 1):
        user = sample_user(rng)

        is_evening = bool(rng.random() < 0.5)
        is_weekend = bool(rng.random() < 0.28)

        x = user.to_feature_vector(is_evening=is_evening, is_weekend=is_weekend)

        arm = agent.select_arm(x)
        reward = env.get_reward(arm, x)
        agent.update(arm, x, reward)

        total_reward += reward
        arm_counts[arm] += 1
        arm_rewards[arm] += reward

        if step % 1000 == 0:
            ctr = total_reward / step
            print(f"step={step:4d} | cumulative CTR={ctr:.4f}")

    print("\n=== Final stats ===")
    print(f"Total steps: {n_steps}")
    print(f"Total reward: {total_reward}")
    print(f"Cumulative CTR: {total_reward / n_steps:.4f}")

    print("\nArm usage:")
    for arm in ARMS:
        count = arm_counts[arm]
        avg_reward = arm_rewards[arm] / count if count > 0 else 0.0
        print(f"  {arm:15s} selected={count:4d}, avg_reward={avg_reward:.4f}")

    print("\nEstimated theta for each arm:")
    for arm in ARMS:
        theta_hat = agent.arms[arm].theta()
        print(f"\n{arm}:")
        print(np.round(theta_hat, 3))

    demo_user = UserProfile(
        age=22,
        gender=1,
        is_student=True,
        works_in_it=False,
        sports_interest=0.3,
        tech_interest=0.8,
        education_interest=0.7,
        entertainment_interest=0.4,
    )
    x_demo = demo_user.to_feature_vector(is_evening=True, is_weekend=False)
    scores = agent.predict_scores(x_demo)

    print("\nDemo user scores:")
    for arm, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {arm:15s} -> {score:.4f}")

    best_arm = agent.select_arm(x_demo)
    print(f"\nRecommended category for demo user: {best_arm}")


if __name__ == "__main__":
    main()
