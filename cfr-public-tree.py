import numpy as np

# DEBUG = True
DEBUG = False

N = 2
A = 2
S = 3
ACTION_DIM = 0

valid_actions = ['p', 'b']
valid_states = ['J', 'Q', 'K']
call_matrix = np.float32([ [ 0,  1,  1],
                           [-1,  0,  1],
                           [-1, -1,  0] ])

fold_matrix = np.float32([ [0,  1,  1],
                           [1,  0,  1],
                           [1,  1,  1]])

def evaluate(h, pi):
    reward = None
    if len(h) >= 2:
        pot = 1 + h.count('b') // 2
        if h[-2:] in ('pp', 'bb'):
            reward = pot * np.matmul(pi[::-1], call_matrix)
        elif h[-2:] == 'bp':
            if h == 'bp':
                fold_player = 1
            else:
                fold_player = 0
            reward = pot * np.matmul(pi[::-1], fold_matrix)
            reward[fold_player] = - reward[fold_player]

    if DEBUG:
        print("--- Evaluate(h = {}) ---".format(h))
        print("  range[0] = {:> 4.2f} {:> 4.2f} {:> 4.2f}".format(*pi[0]))
        print("  range[1] = {:> 4.2f} {:> 4.2f} {:> 4.2f}".format(*pi[1]))
        print("  reward[0] = {:> 4.2f} {:> 4.2f} {:> 4.2f}".format(*reward[0]))
        print("  reward[1] = {:> 4.2f} {:> 4.2f} {:> 4.2f}".format(*reward[1]))
        print("------")
        print("")

    return reward


epsilon = 1e-6
class TreeNode(object):
    def __init__(self, public_state):
        self.public_state = public_state
        self.regret_sum = np.ones( (A, S) ) * epsilon
        self.strategy_sum = np.ones( (A, S) ) * epsilon

    def update(self, regret, pi):
        self.regret_sum += regret.clip(min=epsilon)
        self.strategy_sum += (self.strategy * np.expand_dims(pi, 0))

    @property
    def strategy(self):
        return self.regret_sum / self.regret_sum.sum(ACTION_DIM, keepdims=True)

    @property
    def avg_strategy(self):
        return (self.strategy_sum / self.strategy_sum.sum(ACTION_DIM))

    def isTerminal(self):
        h = self.public_state
        if len(h) >= 2:
            if h[-1] == 'p':
                return True
            elif h[-2:] == 'bb':
                return True
        return False

class CFR(object):
    def __init__(self):
        self._table = {}

    def solve(self, h, pi):
        player = len(h) % 2
        public_state = h
        node = self._table.setdefault(public_state, TreeNode(public_state))

        if node.isTerminal():
            reward = evaluate(h, pi)
            return reward
        else:
            utility = np.zeros( (N, A, S) )
            node_utility = np.zeros( (N, S) )
            for i, a in enumerate(valid_actions):
                if player == 0:
                    new_pi = (pi[0] * node.strategy[i], pi[1])
                else:
                    new_pi = (pi[0], pi[1] * node.strategy[i])
                utility[:, i] = self.solve(h + a, new_pi)
            node_utility[player] = (node.strategy * utility[player]).sum(0)
            node_utility[1 - player] = utility[1-player].sum(0)

            cf_regret = (utility[player] - node_utility[player])

            if DEBUG:
                print("--- Solve(h = {}) ---".format(h))
                print("  Utility[0]:")
                for i, c in enumerate( ('J', 'Q', 'K') ):
                    print('    ',c + h, "P: {:> 4.2f} B: {:> 4.2f}".format(*utility[0, :, i]))
                print("  Utility[1]:")
                for i, c in enumerate( ('J', 'Q', 'K') ):
                    print('    ',c + h, "P: {:> 4.2f} B: {:> 4.2f}".format(*utility[1, :, i]))
                print('------')
                print()

            node.update(cf_regret, pi[player])
            return node_utility

    def train(self, iterations = 1):
        utility = 0
        for i in range(iterations):
            pi = (np.ones(S) / S, np.ones(S) / S)
            utility += self.solve("", pi)
        print("Expectation = ", (utility / (iterations) ).sum(1))

        print("AvgStrategy:")
        for k in sorted(self._table.keys()):
            if self._table[k].isTerminal(): continue
            for i, card in enumerate(['J', 'Q', 'K']):
                print("\t", "infoset[{}]\t= {:.2f} for pass. {:.2f} for bet".format(card + k, *self._table[k].avg_strategy[:, i]))

if __name__ == '__main__':
    cfr = CFR()
    if DEBUG:
        cfr.train(1)
    else:
        cfr.train(10000)
