import random
import pickle
import numpy as np
from bot import PGBot
from poker_pg import *
from pypokerengine.api.game import setup_config, start_poker

#Training hyperparameters
N_GAMES = 10
N_ROUNDS_PER_GAME = 100
SAVE_PATH = 'saved_models/first_test/policy_net_after'

#Game Parameters
MAX_ROUND = 3
INITIAL_STACK = 100
SMALL_BLIND_AMOUNT = 5

#The network
policy = Network()

#Players
bot1 = PGBot('bot_p1', policy)
bot2 = PGBot('bot_p2', policy)

for i in range(N_GAMES):
    config = setup_config(max_round=N_ROUNDS_PER_GAME,
                          initial_stack=INITIAL_STACK,
                          small_blind_amount=SMALL_BLIND_AMOUNT)

    config.register_player(name="bot_p1", algorithm=bot1)
    config.register_player(name="bot_p2", algorithm=bot2)
    game_result = start_poker(config, verbose=0)
    model_save =  SAVE_PATH + str(i * N_ROUNDS_PER_GAME)
    print('Games played:', str(i * N_ROUNDS_PER_GAME))
    policy.save_network(model_save)
