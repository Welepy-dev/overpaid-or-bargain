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

## Data sources (To be decided)
| Source        | Data                                             |
|--------------|---------------------------------------------------|
| Transfermarkt| Transfer Value, age, contract, market value       |
| Understat    | xG, xA, xGChain, xGBuildup                        |

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

## Repository structure (current)

```
.
├── main.py
├── notes.txt
├── pyproject.toml
├── README.md
└── uv.lock
```
1 directory, 5 files