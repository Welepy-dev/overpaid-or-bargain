# Overpaid or bargain? An analysis of the 2026/27 summer transfer window.

In the summer of 2026 we had one of the most expensive windows in history, with such a high volume of investment, every die hard fan of their club would ask himself "was this a good buy"? With this question I came with this project idea:

A data analytics project that evaluates the players of the 2026 summer transfer window, here, we'll see if the value paid is justified with stats based on their perfomance before and after their transfer.

## Questions

1. Based on their statistical profile of the 2025/26 what players were overpaid and underpaid against a estimated "fair price"?
2. What is the current perfomance of the player on their current team?

## Scope and inclusion criteria

- **Leagues:** UEFA's top five (September of 2026): Premier League, LaLiga, Serie A, Bundesliga, Ligue 1.
- **Period:** Summer of 2026 is the window being judged. Summers 2019 to 2025 are used as history to train the fair-price model.
- **Eligible players:** players **bought** by a top-5 club **from another top-5 club** (`only_top5_origin` in `config.yaml`). That gives 1,710 purchases from 2019 to 2026, 222 of them in summer 2026.
- **Excluded:** goalkeepers, loans (including loans with an obligation to buy), free transfers and loan returns.
- **Fee:** the fixed amount only, without bonuses or add-ons.
- **Matches:** league matches only, for every player and every stat (no cups or European games).

## Data sources
| Source        | Data                                             |
|--------------|---------------------------------------------------|
| Transfermarkt| Transfer fee, age, contract, market value before the transfer, position, origin club |
| [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | Public Transfermarkt dump (CC0, frozen July 2026): player details for summers up to 2025 |
| Understat    | xG, xA, xGChain, xGBuildup, shots, key passes (top-5 leagues) |
| Sofascore    | Defensive and possession metrics for every outfield player in the top-5 leagues (league matches only) |
| UEFA ranking | Country coefficients from kassiesa.net (league-strength adjustment) |

FBref lost its Opta data in January 2026 and ClubElo's API requires registration since September 2026, so neither is used by default.

## Methodology (planned)

1. Data collection and integration from various sources
2. Cleaning and integration (one table per purchase, metrics per 90 minutes, market inflation)
3. Exploratory analysis with percentiles and player profiles
4. Regression model on log(fee) to estimate the "fair price", with Transfermarkt market value as a feature and prediction intervals
5. Performance monitoring throughout the 2026/27 season (before/after per 90, cost/performance index, team results with and without the player)

Rules set for the model:
- Purchases with fewer than 900 league minutes in the season before (none at all included) stay out of the price model, but are still followed in phase 5.
- Purchases with an unknown fee stay out of the model and of the ranking.
- Origin-club strength is the club's league position in the season before (ClubElo is no longer available).

## Known limitations

- Small sample size at the start of the season: conclusions about performance will be updated throughout the season 

- Public metrics do not capture the full contribution of a player (positioning, off-the-ball pressure)

- The price of a transfer depends on non-statistical factors (club emergency, clauses, marketing)

## Roadmap

- [x] Phase 0 - Scope definition
- [x] Phase 1 - Data collection (89% of purchases complete; weekly refresh every Monday)
- [x] Phase 2 - Cleaning and integration (one table per purchase, `build` command)
- [ ] Phase 3 - Exploratory analysis
- [ ] Phase 4 - Fair price model (first version: `model` command)
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

Steps: `transfers`, `tm_details`, `understat`, `sofascore_leagues`, `uefa`, `hicp` (euro-area inflation from the ECB, used only by the EDA) (and the optional `sofascore` and `clubelo`). `sofascore` (season stats for players arriving from outside the top 5) was set aside on 3 Oct 2026: about 1,300 players to look up one by one, hours of requests and a high risk of being blocked.
Settings (leagues, seasons, eligibility rules, request interval) live in `config.yaml`.
Every downloaded page is cached in `data/raw/`; tables are written to `data/interim/` as parquet.
A failing step is logged and the next steps still run. Sofascore needs Chromium (set `PECHINCHA_BROWSER` if it is not found).
Player details for summers up to 2025 come from the public dump; only players missing from it are requested from Transfermarkt. Once a summer window has closed, its player pages are cached for good, so the weekly refresh only fetches new players.

### Weekly refresh (scheduled)

The summer window is closed, so transfers and Transfermarkt details are final. Each week only the 2026/27 season stats are refreshed:
`uv run main.py collect --current-only --steps understat sofascore_leagues` (about 5 Understat and ~30 Sofascore requests, under 2 minutes; no Transfermarkt requests).

```bash
./scripts/install_weekly.sh                   # systemd user timer: Mondays at 09:00
./scripts/weekly.sh                           # run it now by hand
systemctl --user list-timers pechincha-weekly.timer
journalctl --user -u pechincha-weekly.service
```

Each run writes `logs/weekly_<date>.log`, ending with the duration and the number of requests per source (pages added to or renewed in the cache).
With `Persistent=true`, a run missed because the PC was off starts at the next login. User timers only run while you are logged in; `loginctl enable-linger $USER` lets them run without a session.
The unit files are in `scripts/systemd/`; to remove the timer: `systemctl --user disable --now pechincha-weekly.timer`.

## Cleaning and integration (phase 2)

```bash
uv run main.py build                      # offline: reads data/interim and the cache, no requests
```

Writes `data/processed/purchases.parquet` (and `.csv`): one row per purchase (summer, player, buying club), 1,710 rows. Each row has the fee, the Transfermarkt profile, the league stats of the season before (Understat and Sofascore, totals and per 90 minutes), the origin club's league position that season, the UEFA rank of the origin league and the model rules. League tables are rebuilt from the cached Understat results into `data/processed/league_tables.parquet`.

- **Bought twice in one summer:** 14 players were bought by two clubs in the same summer (an option exercised and an immediate resale, e.g. Cucurella 2019). Both purchases are real and stay, flagged `bought_twice_in_summer`. Links to Sofascore and Understat belong to the player, so they are joined after removing repeats and no longer duplicate rows.
- **Understat link:** by name, using both the Transfermarkt and the Sofascore name: origin club, origin league, top 5, then last name and nested names at the origin club. Players on loan elsewhere are found at the team Sofascore puts them in that season (Emerson at Betis, Kalulu at Juventus). Every purchase with league minutes is linked. Links that need a human go to `data/manual/understat_links.csv` (fill `understat_id`, run `build` again).
- **Manual Sofascore fixes:** filled rows in `data/manual/sofascore_links.csv` now also override the collected link in `build`, without re-collecting.
- **Model rules:** `in_price_model` leaves out unknown fees (17), purchases with no league minutes the season before (137) and purchases with fewer than 900 (303; `price_model.min_minutes_before` in `config.yaml`); `in_ranking` is the 2026 subset of the model; `in_phase5` keeps every 2026 purchase. `model_exclusion` says why a row is out.
- **Market inflation:** `fee_adjusted` divides the fee by `fee_index`, the median known fee of that summer relative to summer 2026.
- **Checks:** `link_conflict` flags rows where Understat and Sofascore minutes disagree by more than 25% (none after the fixes).
- **League position limits:** ordered by points per game, goal difference and goals scored. Head-to-head tie-breaks and point deductions are not applied (Juventus 2022/23 shows 4th instead of 7th).

## Fair-price model (phase 4)

```bash
uv run main.py build && uv run main.py eda   # tables and percentiles the model reads
uv run main.py model                          # offline: reads data/processed only
```

A ridge regression on log(fixed fee in 2026 prices), trained on the 1,086 price-model purchases of 2019–2025 and applied to the 167 of summer 2026. Features come from an explicit list (`FEATURES` in `src/pechincha/model/fair_price.py`): Transfermarkt value before the transfer, age, league minutes, origin club's league position and origin league, position group, buying league and 16 per-position percentiles (one of each redundant pair from the EDA). Alpha is chosen by predicting each past summer from the others; the 80% interval comes from those out-of-sample errors. A fee above the interval is **overpaid**, below it a **bargain**.

- **Small sample:** about 1,100 purchases for ~30 features. Read each player's verdict with its interval.
- **Market value carries most of the signal:** it alone explains 71% of the variance in log fee on unseen summers; the full model 77%. The stats add little on top, because Transfermarkt values already price them in.
- **Buying league is a feature,** so the fair price is fair for a club in that league (the Premier League premium counts as the market).

Writes `outputs/model/`: `ranking_2026.csv`/`.json` (fee, fair price, interval, verdict and each feature group's part of the prediction), `verdicts_2026.csv` (how many purchases fall below, inside and above the interval, by buying league and position), `model_comparison.csv`, `coefficients.csv`, `cv_by_summer.csv`, seven Plotly charts (`.html` + `.json`) and `summary.md`. Predictions for every model purchase go to `data/processed/fair_price.parquet` (out-of-sample for 2019–2025).

## Repository structure

```
.
├── config.yaml
├── main.py
├── src/pechincha/
│   ├── config.py, http.py, pipeline.py, cli.py
│   ├── build.py         # phase 2: one table per purchase (offline)
│   ├── eda/             # phase 3: percentiles and exploratory charts
│   ├── model/           # phase 4: fair-price model and 2026 ranking
│   ├── report/          # Plotly theme and export
│   └── ingest/          # transfermarkt, tm_dump, understat, sofascore, uefa, hicp, clubelo
├── scripts/             # weekly refresh and its systemd timer
├── tests/               # offline tests with fixtures
├── data/                # raw/ cache, interim/ and processed/ tables (not in git); manual/ links (in git)
├── notes.txt
├── pyproject.toml
└── uv.lock
```
