# Algorithmic Amplification and Content Bias on Twitter/X
### Evidence from the Musk Acquisition — Panel Data & Discrete Choice Models, Spring 2026
**Alex Gamez — The University of Texas at El Paso**

---

## Research Question

Did Elon Musk's acquisition of Twitter (October 27, 2022) causally shift which content categories the platform algorithmically amplifies, relative to what organic user interest would have produced?

---

## Data

| File | Description |
|------|-------------|
| `out/twitter_trending_4yr.csv` | 694,930 unique topic-day observations, Oct 2020–Oct 2024 |
| `out/reddit_category.tsv` | Daily post counts for 16 matched subreddits, Oct 2020–Oct 2024 |
| `out/unique_topics.csv` | Deduplicated topic list with auto-assigned categories |
| `out/top500_topics_4yr.csv` | Top 500 topics for manual review |
| `out/stata_panel.csv` | Long-format DiD panel fed into Stata (`analysis.do`) |
| `out/stata_did_results.csv` | β₃ coefficients exported by Stata for each category |

**Do not commit raw data files** — they are listed in `.gitignore`.

---

## Pipeline

Run everything with one command:
```
run_all.bat
```

| Step | Script | Output |
|------|--------|--------|
| 1 | `twitter_unique.py` | `out/unique_topics.csv`, `out/twitter_category_counts.csv` |
| 2 | `category_analysis.py` | `out/figures/twitter_category_shift.png`, `out/figures/twitter_category_timeseries.png` |
| 3 | `category_did.py` | `out/stata_panel.csv`, `out/category_did_results.csv` |
| 4 | `analysis.do` *(Stata)* | `out/stata_did_results.csv`, `out/figures/event_study_cat/*.png`, `out/figures/parallel_trends/*.png` |
| 5 | `category_plots.py` | `out/figures/parallel_trends/*.png`, `out/figures/event_study/*.png` |
| 6 | `category_demographics.py` | `out/figures/demographics/*.png` |

---

## Identification Strategy

**Difference-in-Differences (DiD), log-deviation model:**

```
log_dev(y_it) = β0 + β1·Twitter_i + β2·Post_t + β3·(Twitter_i × Post_t) + ε_it
```

- `log_dev(y)` = log(y + c) minus pre-period log mean — puts Twitter and Reddit on the same scale
- `Twitter_i = 1` for Twitter observations, 0 for matched Reddit subreddit
- `Post_t = 1` after October 27, 2022
- `β3` = causal treatment effect in log units; convert to % change as `(e^β3 − 1) × 100`
- HC3 heteroskedasticity-robust standard errors; estimated in Stata via `reghdfe`
- **Control:** one matched Reddit subreddit per category (e.g., r/SquaredCircle for Wrestling)

---

## File Reference

| File | Purpose |
|------|---------|
| `category_lexicon.py` | 14-category keyword classifier (CamelCase-aware) |
| `twitter_unique.py` | Deduplicates raw trending data, assigns categories |
| `category_analysis.py` | Category composition bar charts and time series |
| `category_did.py` | Builds DiD panel, exports `stata_panel.csv` |
| `analysis.do` | Stata DiD estimation, event studies, parallel trends |
| `category_plots.py` | Per-category parallel trends and quarterly event study |
| `category_demographics.py` | Audience demographic visualizations |
| `category_subreddit_mapping.py` | Shared subreddit ↔ category mapping |
| `reddit_category_pull.py` | Arctic Shift API pull for matched subreddits (run once) |
| `scraper.py` | Twitter trending scraper (historical pull, run once) |
| `run_all.bat` | Regenerates all Python analysis and figures end-to-end |
| `docs/presentation.tex` | Beamer LaTeX slide deck |

---

## Active Categories (14 in DiD model)

`news_politics` · `wrestling` · `combat_sports` · `sports_nba` · `sports_nfl` · `sports_mlb` · `sports_nhl` · `sports_soccer` · `sports_college` · `sports_womens` · `reality_tv` · `entertainment` · `taylor_swift` · `fandom`

---

## Key Results (β₃, post-acquisition treatment effect)

| Category | β₃ | % vs Baseline | Direction |
|----------|-----|---------------|-----------|
| News & Politics | +0.697 | +101% | Amplified *** |
| Wrestling | +0.641 | +90% | Amplified *** |
| Combat Sports | +0.540 | +72% | Amplified *** |
| Sports — NBA | +0.358 | +43% | Amplified *** |
| Sports — NFL | +0.294 | +34% | Amplified *** |
| Sports — Soccer | +0.291 | +34% | Amplified *** |
| Reality TV | +0.194 | +21% | Amplified *** |
| Entertainment | +0.170 | +19% | Amplified *** |
| Sports — MLB | +0.164 | +18% | Amplified *** |
| Sports — NHL | +0.132 | +14% | Amplified *** |
| Sports — College | +0.044 | +5% | No effect |
| Fandom | −0.008 | −1% | No effect |
| Sports — Women's | −0.544 | −42% | Suppressed *** |
| Taylor Swift | −0.836 | −57% | Suppressed *** |

---

## Requirements

```
pip install pandas numpy matplotlib statsmodels requests
```

Stata `reghdfe` and `regsave` packages required for `analysis.do`.
