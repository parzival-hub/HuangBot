Saved `ExtState` answers of the real Zhanguo server (`GET /api/ext/<gameId>/state`)
with `"yourTurn": true`, one JSON file each. `test_fixtures_from_game.py` replays
every `*.json` file here; with none present it is skipped.
