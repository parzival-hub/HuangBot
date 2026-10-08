"""Tile accounting from public events and one player's private observations.

Unknown hand and discard slots are exchangeable: no opponent strategy is
assumed. Conditioning on known market acquisitions gives a multivariate
hypergeometric belief, rather than consulting the simulator's hidden bag.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
import re

from huang import board
from huang.engine import Color, COLOR_NAMES, TILE_COUNTS

COLORS = tuple(Color)
BY_NAME = {name: color for color, name in COLOR_NAMES.items()}
TILE_BELIEF_FEATURE_SIZE = 37
PLACED = re.compile(r'^P(\d+) placed (\w+) tile at ')
TAKEN = re.compile(r'^P(\d+) took (\w+) from market slot ')
REPLACED = re.compile(r'^P(\d+) replaced (\d+) hidden tile\(s\)$')
SPENT = (
    (re.compile(r'^P(\d+) caused a riot with (\d+) blue tile'), Color.BLUE),
    (re.compile(r'^P(\d+) paid (\d+) green tile'), Color.GREEN),
    (re.compile(r'^P(\d+) committed (\d+) yellow tile'), Color.YELLOW),
    (re.compile(r'^P(\d+) supported war side \d+ with (\d+) red tile'), Color.RED),
)


def draw_color(counts, rng):
    total = sum(counts.values())
    if total <= 0:
        raise ValueError('cannot draw from an empty tile pool')
    target = rng.randrange(total)
    for color in COLORS:
        target -= counts[color]
        if target < 0:
            counts[color] -= 1
            return color
    raise AssertionError('invalid tile pool')


@dataclass(frozen=True)
class TileKnowledge:
    observer: int
    bag_total: int
    hand_sizes: tuple[int, ...]
    own_hand: tuple[int, ...]
    placed: tuple[int, ...]
    spent: tuple[int, ...]
    own_discarded: tuple[int, ...]
    unknown_discarded: int
    known_hands: tuple[tuple[int, ...], ...]
    unknown_pool: tuple[int, ...]
    hidden_discard_sizes: tuple[int, ...] = ()

    @property
    def draw_probabilities(self):
        total = sum(self.unknown_pool)
        return tuple(count/total if total and self.bag_total else 0.0 for count in self.unknown_pool)

    @property
    def expected_bag_counts(self):
        return tuple(self.bag_total*p for p in self.draw_probabilities)

    @property
    def bag_count_variances(self):
        total = sum(self.unknown_pool)
        correction = (total-self.bag_total)/(total-1) if total > 1 else 0.0
        return tuple(self.bag_total*p*(1-p)*correction for p in self.draw_probabilities)

    @property
    def uncertainty(self):
        """Normalized entropy of the next draw; zero for an empty bag."""
        return -sum(p*math.log(p) for p in self.draw_probabilities if p)/math.log(len(COLORS))

    def feature_vector(self):
        """Append-only neural inputs; all values derive from this seat's view."""
        scales = tuple(float(TILE_COUNTS[color]) for color in COLORS)
        features = []
        for counts in (self.placed,self.spent,self.own_discarded,self.unknown_pool,self.expected_bag_counts):
            features.extend(value/scale for value,scale in zip(counts,scales))
        features.extend(self.draw_probabilities)
        features.extend(math.sqrt(value/scale) for value,scale in zip(self.bag_count_variances,scales))
        features.extend((self.unknown_discarded/sum(TILE_COUNTS.values()),self.uncertainty))
        assert len(features) == TILE_BELIEF_FEATURE_SIZE
        return tuple(features)

    def sample(self, rng: random.Random):
        bag,hands,_ = self.sample_world(rng)
        return bag,hands

    def sample_world(self, rng: random.Random):
        """Draw a complete compatible bag/hand allocation without replacement."""
        pool = Counter(dict(zip(COLORS, self.unknown_pool)))
        hands = [Counter(dict(zip(COLORS, known))) for known in self.known_hands]
        hands[self.observer] = Counter(dict(zip(COLORS, self.own_hand)))
        for player, size in enumerate(self.hand_sizes):
            if player != self.observer:
                for _ in range(size-sum(hands[player].values())):
                    hands[player][draw_color(pool,rng)] += 1
        # Unknown discarded colours are latent variables, including the optional
        # historical short-game removals. They are never read from the engine.
        discarded = [Counter() for _ in self.hand_sizes]
        for player,size in enumerate(self.hidden_discard_sizes):
            for _ in range(size):
                discarded[player][draw_color(pool,rng)] += 1
        for _ in range(self.unknown_discarded-sum(self.hidden_discard_sizes)):
            draw_color(pool,rng)
        assert sum(pool.values()) == self.bag_total
        return pool, hands, discarded

    def to_dict(self):
        def named(values):
            return {COLOR_NAMES[color]: values[int(color)] for color in COLORS}
        return {'bag_total':self.bag_total, 'placed':named(self.placed),
                'spent':named(self.spent), 'own_discarded':named(self.own_discarded),
                'unknown_discarded':self.unknown_discarded,
                'expected_bag_counts':named(self.expected_bag_counts),
                'draw_probabilities':named(self.draw_probabilities),
                'bag_count_variances':named(self.bag_count_variances),
                'uncertainty':self.uncertainty}


class TileTracker:
    """Incremental permanent ledger; removing a board tile never replenishes it."""
    def __init__(self, observer):
        self.observer = observer
        self.public_cursor = self.private_cursor = 0
        self.placed = Counter({Color.YELLOW:len(board.CAPITAL_CELLS)})
        self.spent = Counter()
        self.own_discarded = Counter()
        self.unknown_discarded = 0
        self.known_hands = []
        self.hidden_discard_sizes = []

    def observe(self, engine):
        if not self.known_hands:
            self.known_hands = [Counter() for _ in range(engine.players)]
            self.hidden_discard_sizes = [0]*engine.players
            self.unknown_discarded = 24 if engine.short_game else 0
        for event in engine.public_history[self.public_cursor:]:
            match = TAKEN.match(event)
            if match:
                player, name = match.groups()
                self.known_hands[int(player)][BY_NAME[name]] += 1
                continue
            match = REPLACED.match(event)
            if match:
                player, amount = map(int, match.groups())
                if player != self.observer:
                    self.unknown_discarded += amount
                    self.hidden_discard_sizes[player] += amount
                # Only a lower bound survives a concealed exchange.
                for color in COLORS:
                    self.known_hands[player][color] = max(0,self.known_hands[player][color]-amount)
                continue
            match = PLACED.match(event)
            if match:
                player, name = match.groups()
                color, amount = BY_NAME[name], 1
                self.placed[color] += amount
            else:
                for pattern, color in SPENT:
                    match = pattern.match(event)
                    if match:
                        player, amount = map(int, match.groups())
                        self.spent[color] += amount
                        break
                else:
                    continue
            player = int(player)
            self.known_hands[player][color] = max(0,self.known_hands[player][color]-amount)
        self.public_cursor = len(engine.public_history)
        private = engine.private_history[self.observer]
        for event in private[self.private_cursor:]:
            if event.startswith('replaced '):
                counts = tuple(map(int,event[len('replaced '):].split(',')))
                if len(counts) != len(COLORS):
                    raise ValueError('invalid own replacement observation')
                self.own_discarded.update(dict(zip(COLORS,counts)))
        self.private_cursor = len(private)
        # These are the only hand colours that may be observed directly.
        own = engine.hands[self.observer]
        sizes = tuple(sum(hand.values()) for hand in engine.hands)
        market = Counter(color for color in engine.market if color is not None)
        pool = Counter(TILE_COUNTS)
        for color in COLORS:
            pool[color] -= self.placed[color]+self.spent[color]+self.own_discarded[color]+own[color]+market[color]
            for player, known in enumerate(self.known_hands):
                if player != self.observer:
                    pool[color] -= known[color]
        expected = engine.bag_total+self.unknown_discarded+sum(
            size-sum(self.known_hands[player].values()) for player,size in enumerate(sizes) if player != self.observer)
        if any(pool[color]<0 for color in COLORS) or sum(pool.values()) != expected:
            raise ValueError('tile observations do not conserve the starting inventory')
        return TileKnowledge(self.observer,engine.bag_total,sizes,
                             tuple(own[color] for color in COLORS),
                             tuple(self.placed[color] for color in COLORS),
                             tuple(self.spent[color] for color in COLORS),
                             tuple(self.own_discarded[color] for color in COLORS),
                             self.unknown_discarded,
                             tuple(tuple(known[color] for color in COLORS) for known in self.known_hands),
                             tuple(pool[color] for color in COLORS),
                             tuple(self.hidden_discard_sizes))
