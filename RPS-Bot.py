import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import choice

class RPSTrainer:
    def __init__(self):

        self.NUM_ACTIONS = 3
        self.actions = np.arange(self.NUM_ACTIONS)
        
        # __________________________________________
        #           | Rock   | Paper  | Scissors   |
        # Rock      |    1   |   -1   |     1      |
        # Paper     |    0   |    0   |    -1      |
        # Scissors  |   -1   |    1   |     0      |
        # _________________________________________|
        self.actionUtility = np.array([
                    [0, -1, 1],
                    [1, 0, -1],
                    [-1, 1, 0]
                ])
        
        self.regret_sum = np.zeros(self.NUM_ACTIONS)
        self.history_regret_sum = []
        self.strategy_sum = np.zeros(self.NUM_ACTIONS)
        self.history_strategy_sum = []
        self.history_strategy_iteration = []

        self.opponent_regret_sum = np.zeros(self.NUM_ACTIONS)
        self.history_opp_regret_sum = []
        self.opponent_strategy_sum = np.zeros(self.NUM_ACTIONS)
        self.history_opp_strategy_sum = []
        

    def get_strategy(self, regret_sum):
        new_sum = np.clip(regret_sum, a_min=0, a_max=None)
        total_regreat_sum = np.sum(new_sum)
        if total_regreat_sum > 0:
            new_sum /= total_regreat_sum
        else:
            new_sum = np.repeat(1/self.NUM_ACTIONS, self.NUM_ACTIONS)
        return new_sum

    def get_average_strategy(self, strategy_sum):
        average_strategy = [0, 0, 0]
        total_strategy_sum = sum(strategy_sum)
        
        for a in range(self.NUM_ACTIONS):
            average_strategy[a] = strategy_sum[a] / total_strategy_sum
        return average_strategy

    def get_action(self, strategy):
        return choice(self.actions, p=strategy)

    def get_reward(self, my_action, opponent_action):
        return self.actionUtility[my_action, opponent_action]

    def train(self, iterations):

        for i in range(iterations):
            print(f"==================== > Iteration {i}: ")
            
            strategy = self.get_strategy(self.regret_sum)
            opp_strategy = self.get_strategy(self.opponent_regret_sum)
            self.strategy_sum += strategy
            self.opponent_strategy_sum += opp_strategy

            opponent_action = self.get_action(opp_strategy)
            my_action = self.get_action(strategy)

            my_reward = self.get_reward(my_action, opponent_action)
            opp_reward = self.get_reward(opponent_action, my_action)
            print(f"My strategy: {strategy}\nMy action: {my_action}\nMy reward: {my_reward}\nStrategy_sum: {self.strategy_sum}\nRegreat_sum: {self.regret_sum}\n")
            print(f"Opponent strategy: {opp_strategy}\nOpponent action: {opponent_action}\nOpponent reward: {opp_reward}\nOpponent Strategy_sum: {self.opponent_strategy_sum}\nOpponent Regreat_sum: {self.opponent_regret_sum}\n")

            for a in range(self.NUM_ACTIONS):
                my_regret = self.get_reward(a, opponent_action) - my_reward
                opp_regret = self.get_reward(a, my_action) - opp_reward
                self.regret_sum[a] += my_regret
                self.opponent_regret_sum[a] += opp_regret

            self.history_regret_sum.append(self.regret_sum.copy())
            self.history_strategy_iteration.append(strategy)
            self.history_strategy_sum.append(self.get_average_strategy(self.strategy_sum))
            
            # print(self.history_regret_sum)
            # print(self.history_strategy_sum)
            # print("")
            
            self.history_opp_regret_sum.append(self.opponent_regret_sum.copy())
            self.history_opp_strategy_sum.append(self.get_average_strategy(self.opponent_strategy_sum))

            # print(self.history_opp_regret_sum)
            # print(self.history_opp_strategy_sum)
            # print("")

def main():
    trainer = RPSTrainer()
    trainer.train(int(sys.argv[1]))
    target_policy = trainer.get_average_strategy(trainer.strategy_sum)
    opp_target_policy = trainer.get_average_strategy(trainer.opponent_strategy_sum)
    
    df_p1_regret_sum = pd.DataFrame(trainer.history_regret_sum, columns=["Pedra","Papel","Tesoura"])
    df_p1_strategy_iteration = pd.DataFrame(trainer.history_strategy_iteration, columns=["Pedra","Papel","Tesoura"])
    df_p1_strategy_sum = pd.DataFrame(trainer.history_strategy_sum, columns=["Pedra","Papel","Tesoura"])
    
    print(df_p1_regret_sum)
    print()
    print(df_p1_strategy_iteration)
    print()
    print(df_p1_strategy_sum)
    
    # print("Correlation Regret X Strategy:\n")
    # print(df_p1_regret_sum.corrwith(df_p1_strategy_sum, axis=1))
    
    
    print(f"\n========================================\n\t\tResults\n========================================\n")
    print(f'P1 Strategy: {target_policy}')
    print(f'P2 Strategy: {opp_target_policy}')
    
    fig,axis = plt.subplots(3,1, figsize=(15,10))
    
    axis[0].set_title("P1 Regret Sum/iteration")
    sns.lineplot(data=df_p1_regret_sum, ax=axis[0])

    axis[1].set_title("P1 Strategy/iteration")
    sns.lineplot(data=df_p1_strategy_iteration, ax=axis[1])
    
    axis[2].set_title("Best possible strategy P1")
    sns.lineplot(data=df_p1_strategy_sum, ax=axis[2])
    
    plt.show()



if __name__ == "__main__":
    main()
