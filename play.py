from console_player import ConsolePlayer
from pypokerengine.api.game import setup_config, start_poker

from honest_player import HonestPlayer #TODO Update with our bot

#Game Parameters
MAX_ROUND = 3
INITIAL_STACK = 100
SMALL_BLIND_AMOUNT = 5


if __name__ == "__main__":
    config = setup_config(max_round=MAX_ROUND,
                          initial_stack=INITIAL_STACK,
                          small_blind_amount=SMALL_BLIND_AMOUNT)

    config.register_player(name="AI_player", algorithm=HonestPlayer())
    config.register_player(name="human_player", algorithm=ConsolePlayer())
    game_result = start_poker(config, verbose=0)  # verbose=0 because game progress is visualized by ConsolePlayer
