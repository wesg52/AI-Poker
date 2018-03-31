"""Preprocessing script for precomputing hole card win probability based
    on Monte Carlo sampling."""

import pickle
import pypokerengine
from pypokerengine.utils.card_utils import *

def build_compressed_dict():
    """
    Generates a mapping from hole cards to a compressed represenation.
    Returns a dictionary with
        key: c1c2 where c1 is the string representation of c1
        value: (same_suit, card1 rank, card2 rank)
    Note: for each pairing there is no mirror pairing (c2c1) so dict should be
            queried for both orderings.
    """
    d1 = gen_deck()
    d2 = gen_deck()
    hand_to_compressed_hand = {}
    for c1 in d1.deck:
        for c2 in d2.deck:
            if c1 == c2:
                continue
            same_suit = c1.suit == c2.suit
            key = c1.__str__() + c2.__str__()
            value = (same_suit, c1.rank, c2.rank)
            hand_to_compressed_hand[key] = value
    return hand_to_compressed_hand

def generate_hold_card_win_prob(compressed_dict, n_sims=100, n_players=2):
    """
    Perform Monte Carlo sampling to compute win probability.
    Returns dictionary with
        key: (same_suit, card1 rank, card2 rank)  (compressed represenation)
        value: probability of a win
    """
    comp_hands_win_prob = {}
    for key in list(compressed_dict.values()):
        same_suit, rank1, rank2 = key
        if same_suit:
            c1 = Card(suit=2, rank=rank1)
            c2 = Card(suit=2, rank=rank2)
        else:
            c1 = Card(suit=2, rank=rank1)
            c2 = Card(suit=4, rank=rank2)
        prob = estimate_hole_card_win_rate(n_sims, n_players, [c1,c2])
        comp_hands_win_prob[key] = prob
    return comp_hands_win_prob


if __name__ == "__main__":
    NUM_SIMULATIONS = 10
    NUM_PLAYERS = 2

    compressed_dict = build_compressed_dict()
    probability_dict = generate_hold_card_win_prob(compressed_dict,
                                                    NUM_SIMULATIONS,
                                                    NUM_PLAYERS)
    pickle.dump(probability_dict, open('hold_card_win_prob.p', 'wb'))
