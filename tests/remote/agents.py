"""Scripted stand-ins for ``RecurrentModelAgent``."""

from __future__ import annotations

import random


class _Memory:
    def get(self, player):
        raise AssertionError("scripted agents have no memory")


class RandomAgent:
    """Picks a uniformly random legal action and records what it was shown."""

    memory = _Memory()

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.calls = 0
        self.contexts = []

    def select_action(self, context):
        self.calls += 1
        self.contexts.append(context)
        return self.rng.choice(context.legal_actions)

    def reset_episode(self):
        pass


class FirstAgent(RandomAgent):
    """Always the lowest legal action id."""

    def select_action(self, context):
        self.calls += 1
        self.contexts.append(context)
        return min(context.legal_actions)
