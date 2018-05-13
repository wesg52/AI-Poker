import pickle
from console_player import ConsolePlayer
from pypokerengine.api.emulator import Emulator
from bot import PGBot
from pypokerengine.utils.game_state_utils import attach_hole_card_from_deck, attach_hole_card
from pypokerengine.utils.card_utils import gen_cards

STACK = 1000
N_PLAYER = 2
MAX_ROUND = 2
SMALL_BLIND_AMOUNT = 5
ANTE_AMOUNT = 1

if __name__ == '__main__':
    emul = Emulator()
    emul.set_game_rule(N_PLAYER, MAX_ROUND, SMALL_BLIND_AMOUNT, ANTE_AMOUNT)
    console = ConsolePlayer()
    bot = PGBot("bot", pickle.load(open("105980", "rb")))
    emul.register_player("target", bot)
    emul.register_player("baseline", console)
    players_info = {
        "target": {"name": "PGBot" ,"stack": STACK},
        "baseline": {"name": "console", "stack": STACK}
    }
    game_state = emul.generate_initial_game_state(players_info)
    while True:
        game_state, events = emul.start_new_round(game_state)
       
        if events[-1]["type"] == "event_game_finish":
            break
        for player in game_state["table"].seats.players:
            if player.uuid == "target":
                card1 = input("Enter first hole card > ")
                card2 = input("Enter second hold card > ")
                hole_card = gen_cards([card1, card2])
                bot.add_hole_card(hole_card)
                game_state = attach_hole_card(game_state, player.uuid, hole_card)
            else:
                game_state = attach_hole_card_from_deck(game_state, player.uuid)
        game_state, events = emul.run_until_round_finish(game_state)
        if events[0]["round_state"]["round_count"] >= MAX_ROUND:
            break

    print(events[0]["round_state"]["seats"])
