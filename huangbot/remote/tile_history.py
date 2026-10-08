"""Reconstruct the trained tile-belief features from a complete visible ledger."""
from collections import Counter

from huang.engine import Color
from huangbot.tile_belief import TileTracker

from .information import array, cell, color, fail, integer, mapping, required


def tile_knowledge(state, view, seat):
    """Read public events and only this seat's private replacement colours.

    Opponent hands, hidden bag composition and opponent replacement colours
    never enter the ledger. Replaying the complete ledger also supports joining
    a game late, polling gaps, reconnects and rejected decisions.
    """
    history = mapping(required(view, "tileHistory", "view"), "view.tileHistory")
    integer(required(history, "version", "view.tileHistory"), "view.tileHistory.version", minimum=1, maximum=1)
    if required(history, "complete", "view.tileHistory") is not True:
        fail("view.tileHistory.complete", "must confirm all tile events from setup, including removed tiles")
    events = array(required(history, "events", "view.tileHistory"), "view.tileHistory.events")
    public, private = [], []
    for index, event in enumerate(events):
        path = f"view.tileHistory.events[{index}]"
        mapping(event, path)
        kind = required(event, "kind", path)
        player = integer(required(event, "player", path), path + ".player", maximum=state.players - 1)
        if kind == "place":
            name = color(required(event, "color", path), path + ".color")
            space = cell(required(event, "space", path), path + ".space")
            public.append(f"P{player} placed {name} tile at {space}")
        elif kind == "market":
            name = color(required(event, "color", path), path + ".color")
            slot = integer(required(event, "slot", path), path + ".slot", maximum=5)
            public.append(f"P{player} took {name} from market slot {slot}")
        elif kind in ("riot", "pagoda", "revolt", "war", "replace"):
            count = integer(required(event, "count", path), path + ".count", maximum=6)
            if kind in ("riot", "pagoda") and count not in (1, 2):
                fail(path + ".count", "riot/pagoda payments must be one or two tiles")
            if kind == "riot":
                public.append(f"P{player} caused a riot with {count} blue tile(s)")
            elif kind == "pagoda":
                public.append(f"P{player} paid {count} green tile(s) for a pagoda")
            elif kind == "revolt":
                public.append(f"P{player} committed {count} yellow tile(s)")
            elif kind == "war":
                public.append(f"P{player} supported war side 0 with {count} red tile(s)")
            else:
                public.append(f"P{player} replaced {count} hidden tile(s)")
                if player == seat:
                    colors = array(required(event, "colors", path), path + ".colors")
                    if len(colors) != count:
                        fail(path + ".colors", "own replacement colours must match the public count")
                    counts = Counter(color(name, path + ".colors") for name in colors)
                    private.append("replaced " + ",".join(str(counts[c.name.lower()]) for c in Color))
                # Never inspect an opponent's concealed replacement colours.
        else:
            fail(path + ".kind", "unsupported tile event kind")
    state.public_history = public
    state.private_history = [[] for _ in range(state.players)]
    state.private_history[seat] = private
    try:
        knowledge = TileTracker(seat).observe(state)
    except (ValueError, KeyError, IndexError) as error:
        fail("view.tileHistory", f"inconsistent visible tile ledger ({error})")
    visible_board = Counter(state.tiles.values())
    if any(knowledge.placed[int(c)] < visible_board[c] for c in Color):
        fail("view.tileHistory.events", "ledger omits tiles visible on the board")
    return knowledge, (tuple(public), tuple(private))
