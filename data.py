import numpy as np
from pypokerengine.api.emulator import Emulator
from bot import NaiveBot
from pypokerengine.api.game import setup_config, start_poker

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

def data():
    player_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"]
    player_ids = ["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8", "id9"]
    emuluator = Emulator()
    num = np.random.randint(10)
    emulator.set_game_rule(nb_player, final_round, sb_amount, ante)
    emulator.set_game_rule(player_num=num, max_round=10, small_blind_amount=5, ante_amount=1)
    config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)
    players_info = {}
    for p in num:
        players_info.update(player_ids[p], {"name": player_names[p], "stack": 100})
        config.register_player(name=player_names[p], algorithm=NaiveBot(""))
    initial_state = emulator.generate_initial_game_state(players_info)
    game_state, events = emulator.start_new_round(initial_state)
    game_state, _ = emulator.run_until_game_finish(game_state)
    X = np.zeros(num, 112)
    Y = np.zeros(num, 3)
    players = game_state["table"].seats.players
    for p in num:
       player = players.get(player_names[p])
       cards = player.hole_card
       X[p] = NaiveBot("").get_input_vector(game_state, cards)
    return X, Y

