# Closed-loop evaluation plan

**Question:** If the monitor acts while TF++ drives, does it prevent crashes, and what do its false alarms cost?

Status: draft plan, not yet run.

## Goal

Run the monitor **live inside TF++ on Bench2Drive**. When it fires, change how the car drives, then measure:

1. **Benefit:** crashes avoided and the change in driving score.
2. **Cost:** route progress lost, time spent slowed or stopped, new crashes the intervention causes (such as being rear-ended), and new deadlocks.
3. **Near-miss check:** whether intervening during near-miss alarms (Finding 7) helps or does no harm, while truly false alarms carry the real cost.

## Step 0: Shadow mode (do this first)

Run the monitor live **without acting on it** and log its score every step.

- **Parity check:** the live features and score must match what `analysis/monitor_fp.py` produces offline from the same logs. Any mismatch between training and live features would make every later result meaningless.
- **Baseline:** shadow-mode runs are the TF++ baseline for this plan, and they give the live alarm rate per minute.
- **Hardest part:** the box features (`ttc_plan`, `clear_plan`, `conflict_plan`, `corr_dist`) currently come from `src/ew_analysis.extract_route`, which tracks boxes across frames offline. It needs a streaming version that works step by step. The 1 s history features (rolling min, mean and max, plus 0.5 s and 1 s differences) only need a 20-step buffer.

## Step 1: Freeze the monitor

### Model

Keep the model already used in `monitor_fp.py`: scikit-learn `HistGradientBoostingClassifier`.

```python
HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                               min_samples_leaf=200, l2_regularization=1.0,
                               class_weight='balanced', random_state=0)
```

- **Input:** one row per 20 Hz step. Each row holds the current TF++ outputs plus 1 s history features (feature set `tfpp+intent+map+boxes`).
- **Label:** a collision within the next 3 s (`label_3s`).
- **Training data:** only steps where the ego car is moving faster than 2 m/s.
- **Output:** the score is smoothed with an EMA over 0.5 s (span 10) before it is compared with the threshold.

The exact feature list, with how each history feature is computed, goes in the report (professor feedback #1).

### Avoiding leakage and fixing thresholds

- Train on the 3 box seeds with the same **route-grouped 5 folds** as now.
- At run time, each route uses the fold model that never saw that route. Otherwise the closed-loop result is inflated.
- Fix thresholds offline at **1 and 2 false-alarm onsets per minute**, and don't tune them on closed-loop runs.

## Step 2: The intervention

Add a hook in `SensorAgent.run_step` (`carla_garage/team_code/sensor_agent.py`). It adjusts TF++'s target speed before `control_pid_direct` is called, so TF++ still steers. Configure it with environment variables, the same way `UNCERTAINTY_THRESHOLD` is handled.

| Variant | What happens when the alarm is on | Why test it |
|---|---|---|
| **Speed cap (main)** | Cap target speed at a fraction of current speed, such as 0.5×. Hold for at least 1 s and release only when the score falls below a lower threshold | Gentle, so less risk of being rear-ended or stopping in a junction |
| **Hard brake** | Brake fully while the alarm is on | Upper bound on crashes avoided, and worst case for cost |

Log the intervention state in `tfpp_flags` every step: on/off, which variant, and the score.

**Watch for:** interaction with the creep and safety-box logic. In the 4-seed runs, 40 of the 50 "TickRuntime" errors were actually deadlocks, so count new deadlocks separately.

## Step 3: Conditions

Each condition runs on the same routes and seeds, compared pairwise route by route.

| # | Condition | What it shows |
|---|---|---|
| C0 | TF++ with shadow-mode logging | Baseline |
| C1 | Speed-only monitor at the same alarm rate | Whether the GBDT beats the simplest monitor |
| C2 | GBDT without boxes | Value of the temporal features alone |
| C3 | **GBDT with boxes** | Main method |
| C4 | TF++'s own box-in-front check (already logged as `tfpp_bb_in_front`) used to trigger the intervention | Simple rule-based intervention baseline |
| C5 | **Random alarms** with the same rate and duration as C3 | Rules out "any slowing down helps" |
| C6 | **Ground-truth oracle:** intervene when the ground-truth time to collision along the plan is under 1.5 s (the Finding 7 near-miss rule) | Upper bound on what any monitor could achieve |

C5 and C6 matter most. Without C5, a gain could simply come from driving slower. Without C6, there's no way to tell whether a small gain means a weak monitor or an intervention that couldn't help anyway.

## Step 4: Metrics

- **Standard Bench2Drive metrics:** driving score, success rate, route completion, infraction penalty and collisions per km, split into vehicle, pedestrian and static collisions.
- **Cost of the intervention:**
  - interventions per minute and share of time in intervention
  - average speed and route time
  - timeouts and agent blocked
  - new deadlocks
  - **crashes caused by the intervention**, such as being rear-ended within about 3 s of an intervention starting
- **Near-miss split:** at each intervention start, label it justified or truly false using ground truth at that moment, the same as Finding 7. Report crashes avoided and costs separately for each group.
- **Statistics:** pair each run with C0 on the same route and seed, and use route-bootstrap confidence intervals. The baseline driving score varies by about ±0.9 between seeds, and about 18% of routes flip between success and failure from seed to seed, so differences between single runs can't be trusted.

## Step 5: Order of runs

| Phase | Routes | Seeds | Conditions | Purpose |
|---|---|---|---|---|
| A. Shadow and parity | 20 routes | 1 | C0 | Live and offline features match |
| B. Pilot | 44 routes that always fail + 44 matched routes that always pass | 1 | All 7, speed-cap variant | Check effect size, tune hold and release on the pilot only |
| C. Main | All 220 | 3 | C0, C1, C3, C5, C6, plus C3 with hard brake | Final numbers |

**Rough compute estimate:** the last 3-seed × 220-route sweep took about 5 hours on 2 nodes, about 1.7 hours per seed-suite.

- Phase B: about 4–5 hours.
- Phase C: 18 seed-suites, about 30 hours, spread over several days within the 2-node limit.

## What would count as success

- C3 avoids significantly more crashes than C5, the random alarms, at the same alarm rate.
- C3 closes a meaningful share of the gap between C0 and C6, the oracle.
- Near-miss interventions rarely cause new crashes or deadlocks.
- The cost of truly false alarms is reported honestly, as progress lost and route time added per false alarm.

## Open questions for the team

1. Should the speed cap be relative (0.5× current speed) or absolute (such as 3 m/s)?
2. Do we count a collision where the ego car is hit from behind against TF++? Bench2Drive does by default.
3. Who builds the streaming box features? It's the longest single task.
4. The same pipeline can later run on a newer end-to-end model (professor feedback #3), so how general should the hook be?
