import numpy as np

valid_actions = (0, 1, 2)

class RegretMatch(object):
    def __init__(self):
        self._cum_regrets = np.zeros(len(valid_actions))
        self._cum_strategy = np.zeros(len(valid_actions))

    def strategy(self):
        regret_sum = np.sum(self._cum_regrets.clip(min=0))
        if regret_sum <= 0:
            _strategy = np.ones(len(valid_actions)) / len(valid_actions)
        else:
            _strategy = self._cum_regrets.clip(min=0) / regret_sum
        self._cum_strategy += _strategy
        return _strategy

    def average_strategy(self):
        strategy_sum = np.sum(self._cum_strategy)
        if strategy_sum <= 0:
            return np.ones(len(valid_actions)) / len(valid_actions)
        else:
            return self._cum_strategy / strategy_sum

    def update(self, utility, action):
        regret = utility - utility[action]
        self._cum_regrets += regret

    def self_play(self, iterations):
        self.train(iterations, self.strategy)

    def train(self, iterations, dist):
        for i in range(iterations):

            if dist is None:
                action = i % len(valid_actions)
            else:
                action = np.random.choice(valid_actions, p=dist())

            utility = np.zeros(len(valid_actions))
            for a in range(len(valid_actions)):
                res = (a - action + len(valid_actions)) % len(valid_actions)
                if res == 2:
                    utility[a] = -1
                elif res == 1:
                    utility[a] = 1
                else:
                    utility[a] = 0

            action = np.random.choice(valid_actions, p=self.strategy())
            self.update(utility, action)

        print("Regret Match: ")
        print("    Strategy:    {:.2f} {:.2f} {:.2f}".format(*self.strategy()))
        print("    AvgStrategy: {:.2f} {:.2f} {:.2f}".format(*self.average_strategy()))


if __name__ == '__main__':
    print("Case 1: Self-play")
    print("---")
    regret_matcher = RegretMatch()
    regret_matcher.self_play(10000)
    print()

    print("Case 2: Sample from Uniform")
    print("---")
    regret_matcher = RegretMatch()
    regret_matcher.train(10000, lambda: np.ones(len(valid_actions))/len(valid_actions))
    print()

    print("Case 2.1: Sample from 'real' Uniform")
    print("---")
    regret_matcher = RegretMatch()
    regret_matcher.train(10000, None)
    print()

    print("Case 3: Sample from Biased Distribution")
    print("---")
    regret_matcher = RegretMatch()
    regret_matcher.train(10000, lambda: (0.2, 0.4, 0.4))
    print()
