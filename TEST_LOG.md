# Test log
Count of tests run so far (for multiple-testing judgement): see the table. Include earlier work below.

| ID | Date | Idea | Expected | Result (stitched OOS Sharpe unless noted) | Verdict |
| --- | --- | --- | --- | --- | --- |
| E01 | 2026-10-18 | HBA 33 stocks, 72-config walk-forward | baseline | 1.360, median fold 1.282, 88% positive, holdout 0.316 | baseline |
| E02 | 2026-10-18 | 336-config cadence/regime/breadth/target-vol sweep | higher | 1.240 | worse, overfit |
| E03 | 2026-10-18 | MGA, 30 ETFs | diversify | 0.483 | fail |
| E04 | 2026-10-18 | Blend HBA/MGA | smoother | 1.366 at 90/10 vs 1.370 alone | no gain |
| E05 | 2026-10-18 | 86 stocks | breadth | 1.235 | worse |
| E06 | 2026-10-18 | + reversal / low-vol / overnight / all3 | lift | 1.245 / 1.240 / 1.293 / 1.277 | no gain; overnight halves drawdown |
| E07 | 2026-10-18 | Drawdown guard 8% | lower risk | fixed-config Sharpe 1.101 -> 0.934 | hurts |
| T001 | 2026-10-22 | Guard threshold 12/15/20% or off (fixed config, before 2025-08-29, full-history Sharpe) | looser guard keeps Sharpe near unguarded and keeps some drawdown benefit | off 1.153 (maxDD -24.3%); 8% 0.988; 12% 1.125 (-20.7%); 15% 1.066 (-26.8%); 20% 1.122 (-27.3%) | 8% clearly worst; 12% to 20% are within noise of off and not monotonic. Guard off or 12% are equivalent on this evidence. Note: hypothesis was written the same day as the run |

Rows for new ideas: write the first six columns BEFORE running.
