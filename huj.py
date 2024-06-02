import random

from coderone.dungeon.agent import GameState, PlayerState
from coderone.dungeon.game import Game, Recorder
import logging


class RandomAgent:
    ACTION_PALLET = ['', 'u', 'd', 'l', 'r', 'p', '']
    def next_move(self):
        return random.choice(self.ACTION_PALLET)

    def update(self, game_state:GameState, player_state:PlayerState):
        pass

game = Game(row_count=10, column_count=10, max_iterations=1e4, recorder=Recorder())
game.add_agent(RandomAgent(), "agent1")
game.add_agent(RandomAgent(), "agent2")
game.generate_map()

for _ in range(10):
    game.tick(0)

    print(game.stats)
    # for p in game.stats.players.values():
    #     name = "{}{}".format(p.name, '(bot)' if p.is_bot else "")
    #     logging.warning(f"{name} HP: {p.hp} / Ammo: {p.ammo} / Score: {p.score}, loc: ({p.position[0]}, {p.position[1]})")
