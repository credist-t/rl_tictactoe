import random
from collections import defaultdict
from typing import List, Optional, Tuple


class TicTacToe:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> Tuple[int, ...]:
        self.board: List[int] = [0] * 9
        self.current_player: int = 1  # 1 = X, -1 = O
        self.done: bool = False
        self.winner: Optional[int] = None
        return self.get_state()

    def get_state(self) -> Tuple[int, ...]:
        return tuple(self.board)

    def available_actions(self) -> List[int]:
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def check_winner(self) -> Optional[int]:
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for a, b, c in lines:
            s = self.board[a] + self.board[b] + self.board[c]
            if s == 3:
                return 1
            if s == -3:
                return -1
        return None

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool]:
        if self.done:
            raise ValueError("Игра уже завершена.")
        if action not in self.available_actions():
            raise ValueError(f"Недопустимый ход: {action}")

        self.board[action] = self.current_player

        winner = self.check_winner()
        if winner is not None:
            self.done = True
            self.winner = winner
            reward = 1.0
            return self.get_state(), reward, True

        if not self.available_actions():
            self.done = True
            self.winner = 0
            reward = 0.5
            return self.get_state(), reward, True

        self.current_player *= -1
        return self.get_state(), 0.0, False

    def render(self) -> None:
        symbols = {1: "X", -1: "O", 0: "."}
        for i in range(0, 9, 3):
            print(" ".join(symbols[self.board[j]] for j in range(i, i + 3)))
        print()


class QAgent:
    def __init__(
        self,
        alpha: float = 0.2,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99995,
        epsilon_min: float = 0.05,
    ) -> None:
        self.q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def state_key(self, state: Tuple[int, ...], player: int) -> Tuple[Tuple[int, ...], int]:
        return state, player

    def get_q(self, state: Tuple[int, ...], player: int, action: int) -> float:
        return self.q[(self.state_key(state, player), action)]

    def choose_action(self, state: Tuple[int, ...], player: int, actions: List[int], training: bool = True) -> int:
        if not actions:
            raise ValueError("Нет доступных ходов.")

        if training and random.random() < self.epsilon:
            return random.choice(actions)

        q_values = [self.get_q(state, player, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, qv in zip(actions, q_values) if qv == max_q]
        return random.choice(best_actions)

    def update(
        self,
        state: Tuple[int, ...],
        player: int,
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        next_player: int,
        next_actions: List[int],
        done: bool,
    ) -> None:
        old_q = self.get_q(state, player, action)

        if done:
            target = reward
        else:
            next_q = max((self.get_q(next_state, next_player, a) for a in next_actions), default=0.0)
            target = reward + self.gamma * next_q

        self.q[(self.state_key(state, player), action)] = old_q + self.alpha * (target - old_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(agent: QAgent, episodes: int = 200000) -> None:
    env = TicTacToe()

    for episode in range(episodes):
        state = env.reset()

        last_move = {
            1: None,   # (state, player, action)
            -1: None
        }

        while True:
            player = env.current_player
            actions = env.available_actions()
            action = agent.choose_action(state, player, actions, training=True)

            prev_state = state
            next_state, reward, done = env.step(action)

            last_move[player] = (prev_state, player, action)

            if done:
                if reward == 1.0:
                    # текущий игрок победил
                    agent.update(
                        state=prev_state,
                        player=player,
                        action=action,
                        reward=1.0,
                        next_state=next_state,
                        next_player=env.current_player,
                        next_actions=[],
                        done=True,
                    )

                    loser = -player
                    if last_move[loser] is not None:
                        ls, lp, la = last_move[loser]
                        agent.update(
                            state=ls,
                            player=lp,
                            action=la,
                            reward=-1.0,
                            next_state=next_state,
                            next_player=env.current_player,
                            next_actions=[],
                            done=True,
                        )
                else:
                    # ничья
                    agent.update(
                        state=prev_state,
                        player=player,
                        action=action,
                        reward=0.5,
                        next_state=next_state,
                        next_player=env.current_player,
                        next_actions=[],
                        done=True,
                    )
                    other = -player
                    if last_move[other] is not None:
                        os, op, oa = last_move[other]
                        agent.update(
                            state=os,
                            player=op,
                            action=oa,
                            reward=0.5,
                            next_state=next_state,
                            next_player=env.current_player,
                            next_actions=[],
                            done=True,
                        )
                break
            else:
                next_actions = env.available_actions()
                agent.update(
                    state=prev_state,
                    player=player,
                    action=action,
                    reward=0.0,
                    next_state=next_state,
                    next_player=env.current_player,
                    next_actions=next_actions,
                    done=False,
                )
                state = next_state

        agent.decay_epsilon()

        if (episode + 1) % 20000 == 0:
            print(f"Эпизод {episode + 1}/{episodes}, epsilon={agent.epsilon:.4f}")


def human_vs_agent(agent: QAgent, human_symbol: str = "O") -> None:
    env = TicTacToe()
    state = env.reset()

    symbol_to_player = {"X": 1, "O": -1}
    human_player = symbol_to_player[human_symbol.upper()]
    agent_player = -human_player

    print("Нумерация клеток:")
    print("0 1 2")
    print("3 4 5")
    print("6 7 8\n")

    while not env.done:
        env.render()
        player = env.current_player

        if player == human_player:
            actions = env.available_actions()
            while True:
                try:
                    move = int(input(f"Ваш ход ({'X' if human_player == 1 else 'O'}): "))
                    if move not in actions:
                        print("Эта клетка занята или ход недопустим.")
                        continue
                    break
                except ValueError:
                    print("Введите число от 0 до 8.")
            state, _, _ = env.step(move)
        else:
            actions = env.available_actions()
            move = agent.choose_action(state, player, actions, training=False)
            print(f"Ход агента: {move}")
            state, _, _ = env.step(move)

    env.render()
    if env.winner == 0:
        print("Ничья.")
    elif env.winner == human_player:
        print("Вы победили.")
    else:
        print("Агент победил.")


if __name__ == "__main__":
    agent = QAgent(
        alpha=0.2,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.99995,
        epsilon_min=0.05,
    )

    print("Обучение...")
    train(agent, episodes=200000)

    print("\nТеперь можно сыграть против агента.")
    human_vs_agent(agent, human_symbol="O")