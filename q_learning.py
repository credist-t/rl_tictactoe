import random
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Optional, Tuple


#ENVIRONMENT - TICTACTOE BOARD
class TicTacToe:
    """=======================
    Инициализация методов доски
    - очистка, состояние (вектор из 1/0/-1)
    - проверка победителей
    - действия в позиции
    - вывод
    - шаг
    ========================"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board: List[int] = [0] * 9
        self.current_player: int = 1        #1 - AGENT_X, -1 - HUMAN_O
        self.winner: Optional[int] = None
        self.done: bool = False
        return self.get_state()
    
    def get_state(self) -> Tuple[int, ...]:
        return tuple(self.board)
    
    def available_actions(self) -> List[int]:
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def check_winner(self) -> Optional[int]:
        """
        Смотрим суммы по всем возможным позициям
        -> int = +- 1 для победителя если набралось в ряд
        -> None если нет победителя в данной позиции
        """
        lines = [
            (0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8), 
            (0,4,8), (2,4,6)
        ]

        for a,b,c in lines:
            s = self.board[a] + self.board[b] + self.board[c]
            if s == 3:
                return 1
            elif s == -3:
                return -1
        return None
    
    def render(self) -> None:
        symbols = {1: "X", -1: "O", 0: "."}
        for i in range(0, 9, 3):
            print(" ".join(symbols[self.board[j]] for j in range(i, i+3)))

    def step(self, action):
        """
        возвращаем вектор доски, reward, окончена ли партия
        """
        if self.done:
            return self.get_state(), 0.0, True

        if action not in self.available_actions():
            raise ValueError("Недопустимый ход")

        self.board[action] = self.current_player

        winner = self.check_winner()
        if winner:
            self.winner = winner
            self.reward = 1.0
            self.done = True
            return self.get_state(), self.reward, self.done 

        if not self.available_actions():
            self.done = True
            self.winner = 0
            self.reward = 0.5 #draw
            return self.get_state(), self.reward, self.done
        
        self.current_player *= -1
        return self.get_state(), 0.0, False
    

#AGENT - CLASS AGENT
class Agent:
    def __init__(self,
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
        self.epsilon_trace = []

    def get_q(self, state: Tuple[int, ...], player: int, action: int) -> float:
        return self.q[(self.state_key(state, player), action)]

    def state_key(self, state: Tuple[int, ...], player: int) -> Tuple[Tuple[int, ...], int]:
        return state, player
    
    def choose_action(self, state: Tuple[int, ...], player: int, actions: List[int], training: bool = True) -> int:
        if not actions:
            raise ValueError("Нет доступных ходов")
        if training and random.random() < self.epsilon:
            return random.choice(actions)
        
        q_values = [self.get_q(state, player, act) for act in actions]
        best = max(q_values)
        best_actions = [act for act, qva in zip(actions, q_values) if qva == best]
        # print(best_actions)
        return random.choice(best_actions)
    
    def update(
            self,
            state: Tuple[int, ...],   
            action: int,
            player: int,
            reward: float,
            next_state: Tuple[int, ...],
            next_actions: List[int],
            next_player: int,
            done: bool
        ) -> None:
            old_q = self.get_q(state, player, action)
            if done: target = reward
            else:
                next_q = max(self.get_q(state=next_state, player=next_player, action=act) for act in next_actions)
                target = reward - self.gamma * next_q
            
            self.q[(self.state_key(state, player), action)] = old_q + self.alpha * (target - old_q)


    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        

def td_q_learning(agent: Agent, epochs: int = 200_000) -> None:
    env = TicTacToe()

    for episode in range(epochs):
        state = env.reset()
        last_move = {
            1: None,
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
                    agent.update(
                        player=player,
                        state=prev_state,
                        action=action,
                        reward=1.0,
                        next_state=next_state,
                        next_actions=[],
                        next_player=env.current_player,
                        done = True
                    )
                
                    loser = -player
                    if last_move[loser] != None:
                        ls, lp, la = last_move[loser]
                        agent.update(
                            player=lp,
                            state=ls,
                            action=la,
                            reward=-1.0,
                            next_state=next_state,
                            next_actions=[],
                            next_player=env.current_player,
                            done = True
                        )
                else:
                    agent.update(
                        player=player,
                        state=prev_state,
                        action=action,
                        reward=0.5,
                        next_state=next_state,
                        next_actions=[],
                        next_player=env.current_player,
                        done = True
                    )
                    other = -player
                    if last_move[other] != None:
                        os, op, oa = last_move[other]
                        agent.update(
                            player=op,
                            state=os,
                            action=oa,
                            reward=0.5,
                            next_state=next_state,
                            next_actions=[],
                            next_player=env.current_player,
                            done = True
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
                    done=False
                )
                state = next_state
        
        agent.epsilon_trace.append(agent.epsilon)
        agent.decay_epsilon()
        if episode % 10000 == 0:
            print(f"Epoch: {episode}/{epochs}, epsilon: {agent.epsilon:.4f}")

def plot_state_values(agent: Agent, state: Tuple[int, ...], player: int):
    actions = [i for i, cell in enumerate(state) if cell == 0]
    if not actions:
        print("Для этого состояния нет доступных действий")
        return None
    
    q_vals = [agent.get_q(state, player, a) for a in actions]
    v_s = (1-agent.epsilon + agent.epsilon/len(actions)) * max(q_vals) + (agent.epsilon/len(actions)) * (sum(q_vals) - max(q_vals))
    plt.figure(figsize=(7, 4))
    plt.bar([str(a) for a in actions], q_vals, label="Q(s, a)")
    plt.axhline(v_s, linestyle="--", label=f"V(s) = {v_s:.3f}", color='red')

    plt.xlabel("Действие")
    plt.ylabel("Значение")
    plt.title(f"Q(s,a) и V(s) для состояния {state}")
    plt.grid(True, axis="y")
    plt.legend()
    plt.show()

def play(agent: Agent, human_sym: str = "O"):
    env = TicTacToe()
    state = env.reset()
    human_player = 1 if human_sym == "X" else -1
    agent_player = -human_player

    print("Нумерация клеток\n" \
    "0 1 2\n" \
    "3 4 5\n" \
    "6 7 8" \
    "")
    while True:
        while not env.done:
            env.render()
            player = env.current_player
            if player == human_player:
                actions = env.available_actions()
                while True:
                    try:
                        move = input(f"Ваш ход: ({'X' if human_player == 1 else 'O'}): ")
                        if move == "Exit":
                            raise KeyboardInterrupt("Игра завершена.")
                        move = int(move)
                        if move not in actions:
                            print("Неверный ход")
                            continue
                        break
                    except ValueError:
                        print("Введите номер клетки от 0 до 8 включительно.")
                state, _, _ = env.step(move)
            else:   
                actions = env.available_actions()
                move = agent.choose_action(state, agent_player, actions, training=False)
                print(f"Ход агента: {move}")
                state, _, _ = env.step(move)
        env.render()
        if env.winner == 0:
            print("Ничья.")
        elif env.winner == human_player:
            print("Вы победили!")
        else:
            print("Модель победила.")
        env.reset()

def main():
    print("TIC TAC TOE")
    agent = Agent(
        alpha = 0.3,
        gamma = 0.9,
        epsilon = 1.0,
        epsilon_decay = 0.99995,
        epsilon_min = 0.005
    )
    td_q_learning(agent, epochs=200_000)

    #график оценки случайного состояния Q ((s, p), a)
    test_state = (
    0, 0, 1, 
    0, 0, -1, 
    0, -1, 1 
    )
    plot_state_values(agent=agent, state=test_state, player=1)


    plt.plot(range(1, 200_000+1), agent.epsilon_trace, marker='')
    plt.xlabel("Эпизод")
    plt.ylabel("Эпсилон")
    plt.title("Смена epsilon в epsilon-greedy политике по эпохам")
    plt.grid(True)
    plt.show()
    print("Game in process")
    play(agent=agent, human_sym='O')

if __name__ == "__main__":
    main()
