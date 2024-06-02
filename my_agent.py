'''
TEMPLATE for creating your own Agent to compete in
'Dungeons and Data Structures' at the Coder One AI Sports Challenge 2020.
For more info and resources, check out: https://bit.ly/aisportschallenge

BIO: 
<Tell us about your Agent here>

'''
import math
# import any external packages by un-commenting them
# if you'd like to test / request any additional packages - please check with the Coder One team
import random
import time
import numpy as np
import pandas as pd
# import sklearn
from collections import defaultdict
import os
import sys
from coderone.dungeon.agent import GameState, PlayerState
from coderone.dungeon.game import Game, Recorder

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class RandomAgent:
    ACTION_PALLET = ['', 'u', 'd', 'l', 'r', 'p', '']

    def next_move(self):
        return random.choice(self.ACTION_PALLET)

    def update(self, game_state: GameState, player_state: PlayerState):
        pass


class Agent:
    flat_state_size = 26
    action_size = 6
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 0.1
    decay_rate = 0.005
    moves_done = 0

    def __init__(self):
        '''
        Place any initialisation code for your agent here (if any)
        '''
        # check if the qtable exists
        qtable_path = os.path.join(DATA_PATH, 'qtable.npy')
        if os.path.exists(qtable_path):
            self.qtable = np.load(qtable_path)
        else:
            self.qtable = np.zeros((8192, self.action_size))

        self.previous_action = None
        self.previous_state = None

        print("Initialised Agent")
        print("learning_rate: ", self.learning_rate)
        print("discount_rate: ", self.discount_rate)
        print("epsilon: ", self.epsilon)
        print("decay_rate: ", self.decay_rate)

    def next_move(self, game_state, player_state):
        '''
        This method is called each time your Agent is required to choose an action
        If you're just starting out or are new to Python, you can place all your
        code within the ### CODE HERE ### tags. If you're more familiar with Python
        and how classes and modules work, then go nuts.
        (Although we recommend that you use the Scrims to check your Agent is working)
        '''

        ###### CODE HERE ######

        # a list of all the actions your Agent can choose from
        actions = ['', 'u', 'd', 'l', 'r', 'p']

        state = self.get_state_id(game_state, player_state)

        if self.previous_state is not None and self.previous_action is not None:
            new_state = state
            reward = player_state.reward
            print(f'Previous state: {self.previous_state}')
            print(f'Previous action: {actions[self.previous_action]}')
            print(f'New state: {new_state}')
            print(f'Reward: {reward}')
            self.qtable[self.previous_state, self.previous_action] = self.qtable[
                                                                         self.previous_state, self.previous_action] + self.learning_rate * (
                                                                             reward + self.discount_rate * np.max(
                                                                         self.qtable[new_state, :]) - self.qtable[
                                                                                 self.previous_state, self.previous_action])

        self.moves_done += 1

        if self.moves_done > 200:
            np.save(os.path.join(DATA_PATH, 'qtable.npy'), self.qtable)
        # exploration-exploitation tradeoff
        if random.uniform(0, 1) < self.epsilon:
            # explore
            action = np.random.choice(self.action_size-1)
            action += 1
        else:
            # exploit
            action = np.argmax(self.qtable[state, :])

        print(f'Action: {action}')

        self.previous_state = state
        self.previous_action = action

        ###### END CODE ######

        return actions[action]

    def get_state_id(self, game_state, player_state):
        state = [1 if player_state.ammo > 0 else 0]
        state.extend(self.get_closest_enemy(game_state, player_state))
        state.extend(self.get_closest_pickup(game_state, player_state))
        state.extend(self.get_nearby_blocks(game_state, player_state))

        return int(''.join(map(str, state)), 2)

    @staticmethod
    def is_block(game_state, x, y):
        if game_state.is_in_bounds((x, y)):
            entity = game_state.entity_at((x, y))
            if entity in ['ib', 'sb', 'ob']:
                return 1

        return 0

    def get_nearby_blocks(self, game_state, player_state):
        return [
            self.is_block(game_state, player_state.location[0] - 1, player_state.location[1]),
            self.is_block(game_state, player_state.location[0] + 1, player_state.location[1]),
            self.is_block(game_state, player_state.location[0], player_state.location[1] - 1),
            self.is_block(game_state, player_state.location[0], player_state.location[1] + 1)
        ]

    def get_closest_pickup(self, game_state, player_state):
        pickups = game_state.ammo
        pickups.extend(game_state.treasure)
        return self.get_closest_entity(pickups, player_state.location)

    def get_closest_enemy(self, game_state, player_state):
        return self.get_closest_entity(game_state.opponents(player_state.id), player_state.location)

    def get_closest_entity(self, entities, player_position):
        closest_entity = None
        closest_distance = math.inf
        for entity_location in entities:
            distance = self.get_distance(player_position, entity_location)
            if distance < closest_distance:
                closest_distance = distance
                closest_entity = entity_location

        if closest_entity is None:
            return [0, 0, 0, 0]

        x_diff = player_position[0] - closest_entity[0]
        y_diff = player_position[1] - closest_entity[1]

        return [1 if x_diff > 0 else 0, 1 if x_diff < 0 else 0, 1 if y_diff > 0 else 0, 1 if y_diff < 0 else 0]

    @staticmethod
    def get_nearby_entities(game_state, player_state):
        radius = 1  # For a 5x5 grid
        nearby_entities = []
        for dx in range(-radius, radius + 1):
            row = []
            for dy in range(-radius, radius + 1):
                x, y = player_state.location[0] + dx, player_state.location[1] + dy
                if game_state.is_in_bounds((x, y)):
                    entity = game_state.entity_at((x, y))
                    if str(entity).isnumeric():
                        if entity == player_state.id:
                            continue
                        else:
                            entity = 'enemy'
                    row.append(entity if entity is not None else 'empty')
                else:
                    row.append('ib')
            nearby_entities.append(row)
        return nearby_entities

    @staticmethod
    def get_distance(player_position, enemy_location):
        return abs(player_position[0] - enemy_location[0]) + abs(player_position[1] - enemy_location[1])
