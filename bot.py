import random
import pickle
import numpy as np
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

#Precomputed and helper global dictionaries
street_to_num = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
hole_card_win_prob = pickle.load(open('hold_card_win_prob.p', 'rb'))
card_one_hot_encoding = pickle.load(open('card_one_hot_encoding.p', 'rb'))

#Returns 52 dimension one hot encoding of list of card strings
def cards_to_vec(cards):
    vec = np.zeros(52)
    for card in cards:
        vec[card_one_hot_encoding[card]] = 1.
    return vec


class NaiveBot(BasePokerPlayer):

    def __init__(self, name):
        self.player_name = name
        self.state_vec = np.zeros(10)
        self.win_prob = 0.5

    def declare_action(self, valid_actions, hole_card, round_state):
        """Main function for implementing the AI strategy. Currently very
        naive hand coded rules for selecting strategy."""
        input_vector = self.get_input_vector(round_state, hole_card)
        print(input_vector)
        if .8 * input_vector[111] + .2 * input_vector[110] > .5:
            if random.random() < .5 and input_vector[111] > .6:
                action = valid_actions[2]
            else:
                action = valid_actions[1]
        else:
            action = valid_actions[0]
        if isinstance(action['amount'], dict):
            amount = action['amount']['min']
        else:
            amount = action['amount']
        return action['action'], amount

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def get_input_vector(self, state, hole_cards):
        """
        Converts the game state into a feature vector compatible with machine learning algorithms.
        See return for the description of indices.
        """
        input_vec = np.zeros(104)

        hole_card_obj = gen_cards(hole_cards)
        c1 = hole_card_obj[0]
        c2 = hole_card_obj[1]
        hole_vec = cards_to_vec(hole_cards)

        commun_cards = state['community_card']
        commun_vec = cards_to_vec(commun_cards)

        street_vec = np.zeros(4)
        street_num = street_to_num[state['street']]
        street_vec[street_num] = 1.

        round_count = state['round_count']

        pot = state['pot']['main']['amount']
        pot_in_BB = pot / (state['small_blind_amount'] * 2)

        hole_card_key = (c1.suit == c2.suit, c1.rank, c2.rank)
        hole_card_win_p = hole_card_win_prob[hole_card_key]

        win_rate = estimate_hole_card_win_rate(
                    nb_simulation=20,
                    nb_player=self.nb_player,
                    hole_card=hole_card_obj,
                    community_card=gen_cards(commun_cards))

        #TODO: Encoding for last opponent move

        #TODO: Encoding for estimated opponent hand strength

        my_stack = 0
        opp_stack = 0
        for player in state['seats']:
            if player['name'] == self.player_name:
                my_stack = player['stack']
            else:
                opp_stack = player['stack']
        stack_ratio = my_stack / (opp_stack + 1.)

        #TODO: previous win probabililty. Need class prev state vector

        state_vec = np.concatenate((hole_vec,           #0-51
                                   commun_vec,          #52-103
                                   street_vec,          #104-107
                                   [round_count],       #108
                                   [pot_in_BB],         #109
                                   [hole_card_win_p],   #110
                                   [win_rate],          #111
                                   [stack_ratio]))      #112
        #self.state_vec = state_vec
        return state_vec
