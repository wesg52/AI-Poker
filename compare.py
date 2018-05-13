import pickle
from console_player import ConsolePlayer
from bot import PGBot
from pypokerengine.api.game import setup_config, start_poker

MAX_ROUND = 5
INITIAL_STACK = 100
SMALL_BLIND_AMOUNT = 5

winners = []

if __name__ == "__main__":
    for i in range(1980, 200000, 2000):
        for j in range(1980, 200000, 2000):
            if i == j:
                continue
            config = setup_config(max_round=MAX_ROUND,
                                  initial_stack=INITIAL_STACK,
                                  small_blind_amount=SMALL_BLIND_AMOUNT)
                                  bot1 = PGBot('bot1', pickle.load(open(str(i), 'rb')))
                                  bot2 = PGBot('bot2', pickle.load(open(str(j), 'rb')))
                                      config.register_player(name="bot1", algorithm=bot1)
                                      config.register_player(name="bot2", algorithm=bot2)
                                      game_result = start_poker(config, verbose=0)
                                      p1 = game_result['players'][0]
                                      p2 = game_result['players'][1]
                                      if p1['stack'] > p2['stack']:
                                          winners.append(i)
                                              else:
                                                  winners.append(j)
    print(sorted(winners, key = winners.count))
    print(max(set(winners), key = winners.count))
