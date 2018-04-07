import numpy as np
from pypokerengine.api.emulator import Emulator
from honest_player import HonestPlayer
from pypokerengine.api.game import setup_config, start_poker

def vectorize(hole, community):
    """Returns the vector representing the card"""
    cards = hole.extend(community)
    n = len(cards)
    v = np.zeros(n, 16*14)
    cards = hole
    for i in n:
        v[i][14*(cards[i].suit - 1) + cards[i].rank] += 1
    return v

def action(w1, w2, w3, x):
    """w1, w2, w3... : linear classifiers that determine the action
    x: vector representing the hand.
    Returns the action to be made by the agent in the current round."""
    ##if np.dot(w, x) > 0 then ...
    pass

def NBClassfier(X, Y):
    """X: the n * m input matrix 
    Y: n * 1 labels
    Returns the naive bayes classfier w."""
    pass

def self_play(w1, w2, w3, n):
    """Plays the game against it self for n times
    Returns X the n*m input matrix and Y the n * 1 labels."""
    pass 

if __name__ == '__main__':
    m = 52
    w = np.zeros(m)
    player_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"]
    for i in 10:
        emuluator = Emulator()
        num = random.randint(1, 10)
        emulator.set_game_rule(nb_player, final_round, sb_amount, ante)
        emulator.set_game_rule(player_num=num, max_round=10, small_blind_amount=5, ante_amount=1)
        config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)
        for p in num: 
            config.register_player(name=player_names[p], algorithm=HonestPlayer())
        X, Y = self_play(w, w, w, 10**i)
        w = NBClassfier(X, Y)
