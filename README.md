# Overpaid or bargain? An analysis of the 2026/27 summer transfer window.

In the summer of 2026 we had one of the most expensive windows in history, with such a high volume of investment, every die hard fan of their club would ask himself "was this a good buy"? With this question I came with this project idea:

A data analytics project that evaluates the players of the 2026 summer transfer window, here, we'll see if the value paid is justified with stats based on their perfomance before and after their transfer.

## Questions

1. Based on their statistical profile of the 2025/26 what players were overpaid and underpaid against a estimated "fair price"?
2. What is the current perfomance of the player on their current team?

## Scope and inclusion criteria

- **Leagues:** UEFA's top five (September of 2026).
- **Period:** Summer of 2026.
- **Elligible players:** Every player **bought**, excluding loans and Goalkeepers.

## Data sources
| Source        | Data                                             |
|--------------|---------------------------------------------------|
| Transfermarkt| Transfer fee, age, contract, market value before the transfer, position, origin club |
| [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | Public Transfermarkt dump (CC0, frozen July 2026): player details for summers up to 2025 |
| Understat    | xG, xA, xGChain, xGBuildup, shots, key passes (top-5 leagues) |
| Sofascore    | Defensive and possession metrics for every outfield player in the top-5 leagues; season stats for players arriving from other leagues |
| UEFA ranking | Country coefficients from kassiesa.net (league-strength adjustment) |

FBref lost its Opta data in January 2026 and ClubElo's API requires registration since September 2026, so neither is used by default.

## Methodology (planned)

1. Data collection and integration from various sources
2. Cleaning and normalization (metrics per 90 minutes, coins, bonuses)
3. Exploratory analysis with percentiles and player profiles
4. Regression model to estimate the "fair price"
5. Performance monitoring throughout the 2026/27 season

## Known limitations

- Small sample size at the start of the season: conclusions about performance will be updated throughout the season 

- Public metrics do not capture the full contribution of a player (positioning, off-the-ball pressure)

- The price of a transfer depends on non-statistical factors (club emergency, clauses, marketing)

## Roadmap

- [x] Phase 0 - Scope definition
- [ ] Phase 1 - Data collection
- [ ] Phase 2 - Cleaning and integration
- [ ] Phase 3 - Exploratory analysis
- [ ] Phase 4 - Fair price model
- [ ] Phase 5 - Performance in the new team
- [ ] Phase 6 - Dashboard and communication

## Data collection (phase 1)

```bash
uv sync
uv run main.py collect                    # everything: summers 2019-2026
uv run main.py collect --current-only     # weekly refresh: 2026/27 only
uv run main.py collect --steps transfers tm_details   # selected steps
uv run pytest                             # parser and pipeline tests (offline)
```

Steps: `transfers`, `tm_details`, `understat`, `sofascore_leagues`, `uefa`, `sofascore` (and the optional `clubelo`).
Settings (leagues, seasons, eligibility rules, request interval) live in `config.yaml`.
Every downloaded page is cached in `data/raw/`; tables are written to `data/interim/` as parquet.
A failing step is logged and the next steps still run. Sofascore needs Chromium (set `PECHINCHA_BROWSER` if it is not found).
Player details for summers up to 2025 come from the public dump; only players missing from it are requested from Transfermarkt. Once a summer window has closed, its player pages are cached for good, so the weekly refresh only fetches new players.

## Repository structure

```
.
├── config.yaml
├── main.py
├── src/pechincha/
│   ├── config.py, http.py, pipeline.py, cli.py
│   └── ingest/          # transfermarkt, tm_dump, understat, sofascore, uefa, clubelo
├── tests/               # offline tests with fixtures
├── data/                # raw/ cache and interim/ tables (not in git)
├── notes.txt
├── pyproject.toml
└── uv.lock
```
