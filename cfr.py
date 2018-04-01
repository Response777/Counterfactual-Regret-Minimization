import numpy as np

valid_actions = ['p', 'b']
def evaluate(cards, h):
    player = len(h) % 2

    def map_fn(card):
        if card == 'J': return 0
        if card == 'Q': return 1
        if card == 'K': return 2

    is_player_higher = map_fn(cards[player]) > map_fn(cards[1 - player])
    if len(h) >= 2:
        if h[-1] == 'p':
            if h == 'pp':
                return 1 if is_player_higher else -1
            else:
                return 1
        elif h[-2:] == 'bb':
            return 2 if is_player_higher else -2


epsilon = 1e-3
class CFRNode(object):
    def __init__(self):
        self.regret_sum = np.ones(len(valid_actions)) * epsilon
        self.strategy_sum = np.ones(len(valid_actions)) * epsilon

    def update(self, regret, pi):
        self.regret_sum = (self.regret_sum + regret).clip(min=0)
        self.strategy_sum += self.strategy * pi

    @property
    def strategy(self):
        return self.regret_sum / self.regret_sum.sum()

    @property
    def avg_strategy(self):
        return self.strategy_sum / self.strategy_sum.sum()


class CFR(object):
    def __init__(self):
        self._table = {}

    def solve(self, cards, h, pi1, pi2):
        """
          args:
            cards: ground truth
            h: history
            pi: reaching probability

          return:
            utility
        """
        reward = evaluate(cards, h)
        if reward is not None:
            return reward

        player = len(h) % 2
        node = self._table.setdefault(cards[player] + h, CFRNode())

        utility = np.zeros(len(valid_actions))
        for i, a in enumerate(valid_actions):
            if player == 0:
                utility[i] = - self.solve(cards, h + a, pi1 * node.strategy[i], pi2)
            else:
                utility[i] = - self.solve(cards, h + a, pi1, pi2 * node.strategy[i])
        node_utility = np.dot(node.strategy, utility)

        if player == 0:
            cf_regret = pi2 * (utility - node_utility)
            node.update(cf_regret, pi1)
        else:
            cf_regret = pi1 * (utility - node_utility)
            node.update(cf_regret, pi2)
        return node_utility

    def train(self, iterations = 10000):
        utility = 0
        for i in range(iterations):
            utility += self.solve(np.random.permutation(['J', 'Q', 'K']), "", 1, 1)
        print("Expectation = ", utility / iterations)

        print("AvgStrategy:")
        for k in sorted(self._table.keys()):
            print("\t", "infoset[{}]\t= {:.2f} for pass. {:.2f} for bet".format(k, *self._table[k].avg_strategy))

if __name__ == '__main__':
    cfr = CFR()
    cfr.train()
