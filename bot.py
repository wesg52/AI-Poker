import random
import pickle
import numpy as np
from poker_pg import *
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

#Constants
N_FEATURES = 113
N_MOVES = 7

#Precomputed and helper global dictionaries
street_to_num = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
hole_card_win_prob = pickle.load(open('hold_card_win_prob.p', 'rb'))
card_one_hot_encoding = pickle.load(open('card_one_hot_encoding.p', 'rb'))

#Utility functions:

#Returns 52 dimension one hot encoding of list of card strings
def cards_to_vec(cards):
    vec = np.zeros(52)
    for card in cards:
        vec[card_one_hot_encoding[card]] = 1.
    return vec




class PGBot(BasePokerPlayer):

    def __init__(self, name, policy):
        self.player_name = name
        self.n_player = 2
        self.prev_input_vec = np.zeros(N_FEATURES)
        self.hole_card_obj = None
        self.win_prob = 0.5
        self.round_state_vecs = []
        self.round_actions = []
        self.network = policy
        self.stack = 100
        
    def add_hole_card(self, hole_card):
        self.hole_card_obj = hole_card

    def declare_action(self, valid_actions, hole_card, round_state):
        """Main function for implementing the AI strategy. Currently very
        naive hand coded rules for selecting strategy."""
        input_vector = self.get_input_vector(round_state, hole_card)
        action = self.network.choose_action(input_vector)

        self.round_state_vecs.append(input_vector)
        self.round_actions.append(action)

        fold_info = valid_actions[0]
        call_info = valid_actions[1]
        raise_info = valid_actions[2]

        if action == 0: #if fold
            if call_info['amount'] > 0:
                return fold_info['action'], fold_info['amount'] #Fold
            else:
                return call_info['action'], call_info['amount'] #Check
        elif action == 1: #if call
            return call_info['action'], call_info['amount'] #Call/Check
        else:
            if action != 5: #if not All in
                mult = action - 1
                amnt = min(raise_info['amount']['max'], raise_info['amount']['min']*mult)
                return raise_info['action'], amnt #Raise
            else:
                return raise_info['action'], raise_info['amount']['max'] #All in

    def naive_strat(self):
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

    def get_stack(self, round_state):
        stack = 0
        try:
            players = round_state['seats']
            for p in players:
                if p['name'] == self.player_name:
                    stack = p['stack']
        except:
            for p in round_state: #winners
                if p['name'] == self.player_name:
                    stack = p['stack']
        return stack

    def receive_game_start_message(self, game_info):
        self.stack = self.get_stack(game_info)


    def receive_round_start_message(self, round_count, hole_card, seats):
        self.init_vec(hole_card, round_count)

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        cur_stack = self.get_stack(winners)
        if cur_stack == 0:
            reward = -100 + (cur_stack - self.stack)
        else:
            reward = (cur_stack - self.stack) + min(round_state['round_count'],3)
        input_data = np.asarray(self.round_state_vecs)
        input_labels = np.asarray(self.round_actions)
        if len(input_data) > 0:
            self.network.update_weights(input_data, input_labels, reward)
            self.round_state_vecs = []
            self.round_actions = []

    def init_vec(self, hole_cards, round_count):
        if self.hole_card_obj == None:
            hole_card_obj = gen_cards(hole_cards)
            self.hole_card_obj = hole_card_obj
        hole_card_obj = self.hole_card_obj
        c1 = hole_card_obj[0]
        c2 = hole_card_obj[1]
        hole_vec = cards_to_vec(hole_cards)

        hole_card_key = (c1.suit == c2.suit, c1.rank, c2.rank)
        hole_card_win_p = hole_card_win_prob[hole_card_key]

        rnd_count = round_count

        vec = np.zeros(N_FEATURES)
        vec[0:52] = hole_vec
        vec[108] = rnd_count
        vec[110] = hole_card_win_p
        vec[111] = hole_card_win_p

        self.prev_input_vec = vec

    def get_input_vector(self, state, hole_cards):
        """
        Converts the game state into a feature vector compatible with machine learning algorithms.
        See return for the description of indices.
        """
        hole_vec = self.prev_input_vec[0:51]

        commun_cards = state['community_card']
        commun_vec = cards_to_vec(commun_cards)

        street_vec = np.zeros(4)
        street_num = street_to_num[state['street']]
        street_vec[street_num] = 1.

        round_count = state['round_count']

        pot = state['pot']['main']['amount']
        pot_in_BB = pot / (state['small_blind_amount'] * 2)

        hole_card_win_p = self.prev_input_vec[110]
        if street_num > 0:
            win_rate = estimate_hole_card_win_rate(
                        nb_simulation=20,
                        nb_player=self.n_player,
                        hole_card=self.hole_card_obj,
                        community_card=gen_cards(commun_cards))
        else:
            win_rate = hole_card_win_p

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
        self.state_vec = state_vec
        return state_vec
