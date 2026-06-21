# ODD Protocol Description — Tag ABM

**Model:** A Mesa-based agent-based model of agents playing the game of tag
**Source files:** `tag_abm.py` (model), `tag_abm_main_v2.ipynb` (simulation & calibration driver)

This document describes the model following the **ODD protocol** (Overview, Design
concepts, Details) as specified in its second update:

> Grimm, V., Railsback, S. F., Vincenot, C. E., Berger, U., Gallagher, C.,
> DeAngelis, D. L., et al. (2020). *The ODD Protocol for Describing Agent-Based and
> Other Simulation Models: A Second Update to Improve Clarity, Replication, and
> Structural Realism.* Journal of Artificial Societies and Social Simulation, 23(2), 7.
> DOI: 10.18564/jasss.4259

The protocol organises a model description into seven elements: (1) Purpose and
patterns, (2) Entities, state variables, and scales, (3) Process overview and
scheduling, (4) Design concepts, (5) Initialization, (6) Input data, and (7)
Submodels.

A note on how the protocol is applied here. ODD is a guide, not a straitjacket.
The seven-element structure is followed in full because it aids
replication — a reader can locate the scheduling, the initialization, or any
submodel without hunting. The eleven *design concepts* of element 4, however, are
treated proportionately: this is a small model, and several concepts (learning,
prediction, explicit objectives) simply do not apply. Rather than pad a paragraph
for each, the concepts with no real content are stated briefly and together, so
that the concepts that *do* matter here stand out and the model's logic stays
easy to follow. Throughout, the description states what the program does, not
what the model is imagined to mean.

---

## 1. Purpose and patterns

### 1.1 Purpose

The model is a Smaldino-style formal precisification of the verbal predictive-processing
theory of play developed by Andersen, Kiverstein, Miller, and Roepstorff (2023).
Andersen et al. argue that play is behaviour in which an agent, freed from
competing demands, deliberately seeks or creates surprising situations at
sweet-spots of complexity and resolves them — a behaviour experientially
"feel-good" because prediction error is being reduced faster than expected. The
verbal theory is consistent with many tag-game phenomena (self-handicapping,
chasing the slower runner, the disengagement that follows a long monotonous
spell) but, as a verbal theory, it does not uniquely fix any one mechanism.

The purpose of this model is therefore not to fit data but, following Smaldino
(2017), to make the verbal theory concrete enough to scrutinise. The model
formalises one specific mechanism the verbal theory is consistent with: each
agent carries a private *arousal* state in [0, 1], which decays while the agent
is a runner, is restored at tag events, and modulates effective running speed
via a *condition weight* that differs between conditions. The model commits
to: arousal as the inner state being tracked; tag events as the slope-events
that drive its restoration; and the per-condition condition weight as the *sole*
mechanism by which the WIN and FUN ways of playing differ.

These commitments are deliberately stark. Real children's tag is shaped by
spatial layout, friendship, gender, audience, fatigue, learning across rounds,
and many things besides — none of which appears in the model. The model is
"stupid on purpose" in Smaldino's sense: the simplifications are *what makes
the formal investigation possible*. The explicit list of structural commitments
in §1.2 makes the simplifications inspectable, and a verbal model of the same
verbal theory cannot do that.

The two **conditions** the model is run under correspond to the two framings
used in the accompanying empirical study:

- **WIN** — agents play to avoid being tagged for the longest cumulative time;
  the weight on the arousal signal is fixed at zero, so the runner ignores its
  internal state and simply runs at top speed.
- **FUN** — agents play to maximise enjoyment; the weight on the arousal signal
  is fixed at one, so a depleted arousal signal pulls speed down fully (the
  formal analogue of self-handicapping in the verbal theory).

Verbally: in WIN the goal-directed exploit mode dominates; in FUN the
exploratory play mode dominates. The model is run on
each game to produce a time-stamped record of tag events and a per-timestep
record of agent state. These outputs let one ask, of any candidate parameter
set, whether the formalised mechanism produces the directional patterns the
verbal theory commits to.

### 1.2 Structural commitments

The list below makes explicit what the model commits to. Each item is a
structural choice; together they fix the scope of what the model can and
cannot produce. Anything outside this list is, by construction, beyond the
model's reach — and that is the point. Following Smaldino (2017), a stupid
model is useful precisely because the things it leaves out are *visibly*
left out.

- **Roles and target selection.** The tagger always runs at maximum speed
  regardless of condition. The tagger's target-selection rule is
  condition-invariant — a softmax over runner speeds, biased toward slow
  runners, in both FUN and WIN. Only runners modulate behaviour by condition.
- **Arousal as the inner state.** Arousal is a private scalar per agent in
  [0, 1], decaying linearly when the agent is a runner and restored at tag
  events. The newly-tagged player and the bystander runners receive separate
  restorations, with the new tagger's kept the larger of the two.
- **Condition weight as the only mechanism.** The per-condition condition
  weight on arousal is the *only* parameter that differs between FUN and WIN,
  and it is fixed at its extremes, one in FUN and zero in WIN. Every other
  process is condition-invariant.
- **Heterogeneity.** Players differ only in `max_speed_i`, drawn from a fixed
  Gaussian per group. There is no per-player weight, decay rate, skill,
  preference, or friendship.
- **No learning, no fatigue.** Agents do not learn across rounds, do not
  fatigue, do not get bored of being tagger, and do not remember previous
  rounds. Repeated games played by the same group are statistically
  equivalent draws.
- **Space is 1-D.** Only chase distance is represented. There are no corners,
  no walls, no field shape, no clustering, no proximity to other runners.
- **No social dimension.** No social preferences, gender, friendship,
  alliances, shared rewards, audience effects, or empathy for the tagger's
  frustration.
- **Tag events are the only arousal source.** Near-misses, deception, taunting,
  cartwheels, rule invention, and watching others all produce no arousal
  recovery in the model.
- **Uniform condition uptake.** All agents perfectly and identically accept
  the condition framing. There is no individual variation in how strongly an
  agent treats the round as FUN or WIN.
- **Arousal is not a reportable state.** Arousal modulates behaviour but is
  not assumed to be introspectible. The model produces no quantity intended
  to correspond to a self-report of "I had fun".

### 1.3 Patterns and parameterisation

The model has many parameters, and they could in principle be tuned until the
model produces whatever pattern is wanted. To avoid that, the calibration does
not tune toward the result. The two conditions are fixed at their extreme forms,
`CONDITION_WEIGHT_FUN = 1` (self-handicapping fully on) and
`CONDITION_WEIGHT_WIN = 0` (fully off), and every other parameter is either fixed
by something outside the model or swept across a plausible range and left to
vary. The FUN-over-WIN tag difference, and whether a WIN game stalls, are then
*outputs* to be observed, not requirements built into the search. Following
Smaldino (2017), the model is left free to fail: nothing forces FUN to out-tag
WIN, and nothing keeps a WIN game alive.

The only requirement a parameter set must meet is a plausibility check, applied
to both conditions alike: a set's average game, in either condition, may not
exceed 60 tags. A tag every two seconds in a 120 s round is faster than any real
game of tag, so sets that average more are discarded as physically implausible.
There is no requirement that FUN exceed WIN, and no minimum tag count, so a WIN
game is allowed to fall to zero tags.

Each parameter set is judged on 100 paired games. For each of the 100, a group of
five players is drawn from the speed distribution, and one FUN game and one WIN
game are played with the same players, so the two conditions are always paired on
the same individuals and differ only in the condition weight. A set's FUN and WIN
tag rates are the averages over its 100 games in each condition.

Two restrictions are imposed at sampling time, both by redrawing sets that
violate them. The three chase distances are kept in the order
`CATCH_DISTANCE < STARTING_DISTANCE < GIVE_UP_DISTANCE`, so a chase can begin and
can be abandoned but never the reverse. And the new tagger's arousal restoration
is kept above the runners', `AROUSAL_RESTORATION_TAGGER > AROUSAL_RESTORATION_SWITCH`,
encoding the theory's claim that becoming the active chaser is a larger jolt of
engagement than a bystander gets from a tag (Andersen et al., 2023).

The calibration is therefore theory-explication, not data-fitting (Smaldino,
2017): it asks whether the formalised mechanism, run across a wide region of
plausible parameter values, produces the directional pattern the verbal theory
commits to, and what within that region drives it. The parameter ranges and the
reasoning behind each are set out in §6 and Appendix A. A sensitivity analysis
(Spearman rank correlation of each swept parameter against the tag counts,
`tag_abm_main_v2.ipynb`) shows which parameters move the outcome: it captures
monotonic main effects, and because the parameters are sampled independently each
correlation isolates one parameter's effect, though interactions between
parameters are not captured.

---

## 2. Entities, state variables, and scales

The model contains two kinds of entity: the **player agent** and the **game model**.
A higher-level **experiment runner** orchestrates many game models but is not itself
an agent.

### 2.1 Player agent (`PlayerAgent`)

A player agent represents one individual in a tag game. State variables:

| Variable | Type | Meaning |
|---|---|---|
| `unique_id` | int | Player index within its group, in the range [1, N]. Constant. |
| `max_speed_i` | float | The agent's maximum (intrinsic) running speed in m/s. A fixed trait, drawn at initialization and never changed. |
| `current_speed_i` | float | The agent's effective running speed this timestep in m/s. Derived each step from `max_speed_i`, the agent's role, arousal, and the condition weight. |
| `current_arousal_i` | float | The agent's arousal level, bounded in [0, 1]. Starts at 1.0. Decays while the agent is a runner; is restored at tag events. |
| `is_tagger` | bool | `True` if the agent is currently the tagger, `False` if a runner. Exactly one agent in a game is the tagger at any time. |

### 2.2 Game model (`TagModel`)

A game model represents a single, complete game of tag played by one group under
one condition. State variables:

| Variable | Type | Meaning |
|---|---|---|
| `group_id` | int | Identifier of the group playing this game. Groups are numbered. |
| `condition` | str | `'WIN'` or `'FUN'`. |
| `condition_weight` | float | The condition weight applied to the arousal signal when computing runner speed in this game. Set from `CONDITION_WEIGHT_FUN` or `CONDITION_WEIGHT_WIN` according to `condition`. See §7.2. |
| `current_time` | float | Elapsed game time in seconds. |
| `current_step` | int | Number of timesteps elapsed. |
| `game_over` | bool | Whether the game has ended. |
| `current_target` | agent or None | The runner currently being chased, or `None` if no target is assigned. |
| `chase_distance` | float | The current distance in metres between tagger and target. |
| `chase_timer` | float | Time in seconds the tagger has spent on the current target. |
| `immune_previous_tagger_id` | int or None | The id of the agent who was tagger immediately before the current tagger. This agent cannot be selected as a target. |
| `immune_abandoned_id` | int or None | The id of the runner most recently abandoned by the tagger. This agent cannot be selected as a target. |
| `event_rows` | list | The accumulating event log for this game (see §4.11 / §7). |
| `agent_records` | list | The accumulating per-agent, per-timestep state log. |
| `schedule` | scheduler | A Mesa `BaseScheduler` holding the player agents (used here only as the agent registry — see §3.3). |

The game model also stores, as fixed attributes for the duration of a game, the
parameters that govern its submodels: `delta_time`, `arousal_decay_rate`,
`arousal_restoration_tagger`, `arousal_restoration_switch`,
`softmax_temperature`, `starting_distance`, `catch_distance`, `give_up_distance`,
`give_up_time`, `pursuit_noise`, and `game_duration`.

### 2.3 Scales

- **Time.** Time is discrete. One timestep advances the game clock by `DELTA_TIME`
  seconds (0.5 s in the calibrated configuration). A game runs for `GAME_DURATION`
  seconds (120 s), i.e. `GAME_DURATION / DELTA_TIME` steps.
- **Space.** Space is not represented as a 2-D arena. Spatial relationships are
  abstracted to a single scalar — the `chase_distance` between the tagger and the
  current target — measured in metres. There are no coordinates and no other
  spatial structure.
- **Population.** Each game involves one group of `N_PLAYERS` agents (5). The
  calibration evaluates each parameter set on 100 paired games: for each, a group
  of five players is drawn and plays one FUN and one WIN game with the same
  players.

---

## 3. Process overview and scheduling

### 3.1 Scheduling at the experiment level

The experiment runner (`run_experiment`) iterates over groups, and within each group
over conditions. For each (group, condition) pair it constructs one game model and
runs it to completion before moving to the next. Max speeds are drawn once per group
and reused across that group's conditions, so the same individuals (with the same
intrinsic speeds) play under both WIN and FUN.

The calibration driver in `tag_abm_main_v2.ipynb` uses a related but distinct
schedule: each parameter set is evaluated on 100 paired games. For each, a group
of five speeds is drawn and one WIN and one FUN game are played with it, so the
two conditions share the same players, and each game is given its own random
seed.

### 3.2 Scheduling within a game

A game runs as a fixed loop of timesteps. Each timestep, the game model executes the
following sequence (`TagModel.step`):

1. **Target selection.** If the tagger currently has no target, one is selected from
   the eligible runners (Submodel §7.3), `chase_distance` is set to
   `STARTING_DISTANCE`, and `chase_timer` is reset to 0.
2. **Chase update.** `chase_distance` is updated from the speed difference between
   tagger and target plus a noise term (Submodel §7.4).
3. **Timer update.** `chase_timer` is increased by one `delta_time`.
4. **Exit-condition evaluation.** In order: if `chase_distance` is at or below
   `CATCH_DISTANCE`, a **catch** occurs — roles switch and the tag is logged
   (Submodel §7.5); else if `chase_distance` is at or above `GIVE_UP_DISTANCE`, the
   chase is **abandoned** (Submodel §7.6); else if `chase_timer` is at or above
   `GIVE_UP_TIME`, the chase is **abandoned**.
5. **Arousal — time decay.** Every runner loses arousal to the passage of time
   (Submodel §7.1). The tagger is exempt.
6. **Arousal — tag restoration.** *Only if a catch occurred this step*, every
   agent's arousal is restored toward 1.0 (Submodel §7.7).
7. **Speed update.** Every agent recomputes its effective speed (Submodel §7.2),
   **once**, from its now-final arousal.
8. **Clock advance and data collection.** `current_step` and `current_time` advance;
   the state of every agent is recorded.
9. **Game-end check.** If `current_time` has reached `GAME_DURATION`, the game ends.
   No end-of-game row is logged — the game simply stops.

The ordering of steps 5–7 is deliberate and reflects the intended causal sequence:
an agent's arousal changes first because time has passed (step 5), then — if a
tag has occurred — because of the tag event (step 6); only then, with arousal
fully resolved, is speed computed (step 7). Speed is computed exactly once per tick;
there is no second speed calculation.

### 3.3 State updating

Each agent's arousal depends only on that agent's own previous arousal and the
elapsed time; each agent's speed depends only on that agent's own arousal, role,
and the (constant) condition weight. No agent reads another agent's state during
these updates. Update order is therefore irrelevant: processing the agents in any
order, or updating them all "simultaneously", yields identical results — the
synchronous/asynchronous distinction does not arise for this model. The model holds
its agents in a Mesa `BaseScheduler`, but uses it only as an agent registry; the
per-step logic is driven explicitly by `TagModel.step` in the order of §3.2. The one
truly between-agent process — target selection, which reads every runner's speed
— together with the chase is handled once per timestep at the model level, not per
agent.

---

## 4. Design concepts

### 4.1 Basic principles

The model rests on the principle that the *roles* in tag (one chaser, several
evaders) and the *transfer* of those roles on a successful catch are the engine
of the dynamics. Layered onto this is *arousal*: a private scalar state per
agent, depleting when one is a runner, restored on tag events, and modulating
runner speed. The central design idea is drawn from predictive processing: the
condition does not change *what* arousal is or *how* it changes, only the
**condition weight** placed on the arousal signal when it is used to set
behaviour. A high weight means the signal shifts the runner's
speed (FUN, exploratory mode); a low weight means the runner ignores its
internal state and simply runs at top speed (WIN, compete
mode). The condition mechanism is therefore reified as a single per-condition
parameter, `CONDITION_WEIGHT_FUN` or `CONDITION_WEIGHT_WIN`, applied
identically to every agent.

This is not the only formalisation of Andersen et al.'s verbal theory; it is
*one* formalisation, sharp enough to investigate. The thing being investigated
is whether this minimal weighting mechanism, with everything else
held constant, can produce the directional patterns the verbal theory commits
to. The structural commitments listed in §1.2 say what the model gives up to
achieve that sharpness.

### 4.2 Emergence

The key model outputs — the number of tags in a game and the distribution of
tags across players — emerge from the repeated interaction of target selection,
chasing, catching, and arousal dynamics. They are not imposed. In particular,
the difference in tagging activity between FUN and WIN emerges from a single
parameter difference (the condition weight) propagating through the speed and
chase submodels: in FUN, depleted arousal pulls runner speed down, the
tagger's relative-speed advantage grows, catches happen more often, restoring
arousal to all players and resetting the cycle; in WIN, arousal still depletes
but barely affects speed, so catches happen at the rate set by chase geometry
alone — which in practice means at the rate set by the pursuit noise: with
`PURSUIT_NOISE` set to zero, WIN collapses to roughly one tag per round while
FUN persists. The model thus locates the two conditions' tag rates in different
causes: accident in WIN, self-handicap in FUN. The distribution of tags across
players is not set anywhere; it is a
by-product of softmax target selection interacting with the heterogeneous
intrinsic speeds.

### 4.3 Adaptation, sensing, and interaction

**Adaptation.** Agents have one adaptive behaviour, exercised only by the tagger: the
choice of *which* runner to chase. The tagger adaptively biases its choice toward
slower runners (Submodel §7.3). Runners do not adapt; their arousal decay is
automatic, not a decision.

**Sensing.** The tagger senses the current speed of every eligible runner (used for
target selection) and the current `chase_distance` to its target (used to detect
catches and abandonment). Runners sense nothing about other agents. All sensing is
assumed perfect and without cost or delay.

**Interaction.** Interaction is mediated entirely through the chase. The tagger and
one target interact each step via the `chase_distance` variable: the gap closes or
widens according to their speed difference. A catch is a direct interaction that
switches roles. Arousal restoration at a catch is a group-wide interaction: the
event affects every agent's arousal, not only the two directly involved. Two
immunity slots constrain interaction by making certain runners temporarily
un-targetable (see §4.6).

### 4.4 Objectives, learning, and prediction

These three ODD concepts are deliberately empty in this model — a fact best
stated plainly rather than padded. No agent evaluates an explicit **objective**
function: the tagger's preference for slower prey is an *implicit* objective,
encoded directly as a probability rule rather than computed. There is no
**learning**: no agent changes its decision rules or trait values with
experience; intrinsic speed is fixed, the tagger's selection rule is identical
across rounds and conditions, and an agent's behaviour in one game is
indistinguishable in expectation from its behaviour in the next. There is no
**prediction**: target selection and chasing use only current speeds and the
current gap, with no representation of future states.

It is worth saying, given the model's predictive-processing motivation, that
the *absence* of explicit prediction in the model is also a commitment. The
verbal PP theory casts agents as prediction-error-minimisers; the formal model
makes no claims about how the agents compute. Arousal here is the *behavioural
trace* of a notional underlying prediction-error dynamic, not the dynamic
itself. Whether a richer model that represented agents' generative models
explicitly would produce different patterns is a question this model cannot
answer.

### 4.5 Stochasticity

Stochasticity enters the model in four places:

1. **Intrinsic speeds** are drawn once per group from a Gaussian
   (`MU_MAX_SPEED`, `SIGMA_MAX_SPEED`), clipped at a lower bound of 0.5 m/s, so that
   groups differ in their composition.
2. **First-tagger assignment** is uniform random over the group's players.
3. **Target selection** is a softmax-weighted random draw over eligible runners,
   biased toward slower runners; `SOFTMAX_TEMPERATURE` controls how random it is.
4. **Pursuit noise** — each step the chase distance receives an additive Gaussian
   perturbation with standard deviation `PURSUIT_NOISE`, representing agility,
   direction changes, terrain, and tactical variation that belong to the
   *interaction* rather than to either agent. The perturbation is per-tick, not
   scaled by the square root of the timestep, so the dynamics are tied to the
   fixed `DELTA_TIME`; changing the timestep would change the model, not just
   its resolution. This noise term is load-bearing in WIN: it is what produces
   the occasional catch when runner speeds barely differ from the tagger's
   (see §4.2).

All randomness is drawn from seeded NumPy generators: a master generator seeds each
game, and each game has its own generator, so runs are fully reproducible.

### 4.6 Collectives

The **group** is a collective: a fixed set of agents who play together and share a
pool of intrinsic speeds. Groups do not interact with one another. Within a game,
the set of {tagger, runners} is a transient collective whose membership is
reshuffled at every catch. Two collective-level constructs constrain play: the
*previous-tagger immunity* (the agent who was just tagger cannot be re-targeted, and
stays immune for the whole of the current tagger's turn) and the *abandoned-runner
immunity* (the runner most recently given up on cannot be re-targeted until the
tagger abandons someone else).

### 4.7 Observation

Two data products are collected for analysis:

- **Event log** (`events_df`) — one row per logged event, with columns `group`,
  `condition`, `runner_tagged_id`, and `timestamp`. The first row of each game
  (`timestamp = 0`) records the randomly assigned initial tagger; it is a marker,
  not a real tag, and is dropped before analysis. Every subsequent row is a real
  tag. **No end-of-game row is logged** — the game simply stops at `GAME_DURATION`,
  and the last logged row is a genuine tag. `runner_tagged_id` is a
  cross-group-unique string of the form `"<group>_<player>"` (e.g. `"0_5"`); because
  groups are numbered, this id never collides with real experimental data, where
  groups are lettered.
- **Agent-state log** (`agent_df`) — one row per agent per timestep, with columns
  `group`, `condition`, `step`, `time`, `player`, `current_arousal_i`,
  `current_speed_i`, `max_speed_i`, and `is_tagger`. Collection is explicit (the
  model appends a record per agent each step).

Before the calibration driver computes any metric, each game's event log is cleaned
exactly as the real experimental data is: the first row (the random initial-tagger
marker) is dropped, and the final row is kept and counted as a genuine tag — only
its time-as-tagger *duration* is treated as undefined, since the game ended before
that spell finished.

---

## 5. Initialization

A simulation is initialized as follows.

**Per group**, before any game is played, the group's `N_PLAYERS` intrinsic speeds
are drawn from a Gaussian distribution with mean `MU_MAX_SPEED` and standard
deviation `SIGMA_MAX_SPEED`, then clipped to a minimum of 0.5 m/s. The same speed
vector is reused for the group's paired FUN and WIN games, so the group's
composition is held constant across the two conditions.

**Per game** (`TagModel.__init__`):

- The game clock (`current_time`) and step counter (`current_step`) are set to 0;
  `game_over` is `False`; the event log is empty.
- The condition determines `condition_weight`: `CONDITION_WEIGHT_WIN` if the
  condition is WIN, `CONDITION_WEIGHT_FUN` if FUN.
- Both immunity slots are empty; `current_target` is `None`; `chase_distance` and
  `chase_timer` are 0.
- One `PlayerAgent` is created per intrinsic speed, with `unique_id` running from 1
  to `N_PLAYERS`. Each agent starts with `current_arousal_i = 1.0` (the ceiling),
  `current_speed_i = max_speed_i`, and `is_tagger = False`.
- A first tagger is chosen uniformly at random among the players and its `is_tagger`
  flag is set. This assignment is logged as an event row at `timestamp = 0` (a
  marker, dropped before analysis).
- The initial state of every agent is recorded before any step runs.

The model state at `t = 0` is therefore: every agent at full arousal, one
randomly chosen tagger, no active chase. Initialization is otherwise fully
determined by the parameters and the seeds; there is no warm-up period.

---

## 6. Input data

The model does **not** use external time-series input data to represent driving
environmental processes. All quantities are generated internally from parameters and
random number streams.

The model's only "inputs" are its parameter set (a Python dictionary) and the
random seeds. The full set is listed in Appendix A. The parameters fall into two
classes. *Fixed parameters* each take a single value: because the experiment
fixes it (five players, 120-second games), because it is a modelling choice
(`DELTA_TIME`, since the pursuit noise is defined per step, see §4.5), because
theory fixes it (the condition weights, one in FUN and zero in WIN), or because
it is anchored to external data (the speed distribution, see below). *Swept
parameters* have no single defensible value, so each is given a range and sampled
uniformly: the chase geometry (`STARTING_DISTANCE`, `CATCH_DISTANCE`,
`GIVE_UP_DISTANCE`, `GIVE_UP_TIME`, `PURSUIT_NOISE`), bounded by what is possible
on a playground, and the arousal and targeting parameters (`AROUSAL_DECAY_RATE`,
the two restorations, `SOFTMAX_TEMPERATURE`), each swept across the full span over
which it has any effect. Appendix A gives every range and the reasoning, and the
two sampling restrictions (the distance ordering and the tagger restoration above
the runners') are described in §1.3.

The model's outputs — the event log and the agent-state log — are described in
§4.7.

**Where the speed distribution comes from.** Each agent's intrinsic maximum
speed is drawn from a Gaussian with mean `MU_MAX_SPEED` = 6.0 m/s and standard
deviation `SIGMA_MAX_SPEED` = 0.5 m/s, clipped at a 0.5 m/s floor. Both values
are anchored to straight-line sprint data for children in the experiment's age
band (11 to 13 years). Vandoni et al. (2024) report 30 m sprint times for a
large sample of Italian 11 to 13 year olds; converted to average speed over the
distance, these run from roughly 5.6 m/s at age 11 to 6.1 m/s at age 13, with
boys marginally faster than girls. A mean of 6.0 m/s sits in the middle of that
band. It is worth stressing that `max_speed_i` is a maximum, not an average: the
tagger runs at exactly `max_speed_i` and runners are scaled below it by arousal
and the condition weight, so the realised in-game average emerges from the model
and is lower than 6.0. The 30 m figures are themselves averages over a
standing-start run, so they slightly understate flying peak velocity, which is
closer to 6.5 to 7.5 m/s for the fastest 13 year olds. A mean of 6.0 m/s is
therefore a conservative ceiling rather than a true sprint maximum.

For the spread, the same source gives a within-group standard deviation on 30 m
time of about 0.4 to 0.5 s, which converts to roughly 0.45 m/s in speed; pooling
the two sexes and the three ages adds a little more, giving an empirical
standard deviation near 0.5 m/s, a coefficient of variation of about 8 percent.
`SIGMA_MAX_SPEED` = 0.5 m/s matches this directly.

**Where the swept ranges come from.** The swept parameters have no single
defensible value, so each is given a range rather than a point and the
calibration samples across it. The ranges are set by behaviour, found by varying
each parameter on its own while holding the rest at a representative set, and
they fall into two kinds. Parameters that move the tag count are bounded by the
span over which they do so. Below the floor the mechanism is effectively off: at
an arousal decay rate of 0.005 a runner barely depletes over a 120 s game, so
self-handicapping is negligible and FUN behaves like WIN. Above the ceiling the
effect either saturates, as with the restorations, where a value of 3 already
restores about 95 percent of the gap to full and larger values change nothing,
or the tag rate runs past the 60-tag plausibility ceiling and the set is
discarded, as very low restoration or very high decay push FUN above 60. The
range therefore brackets the active band. Parameters that barely move the tag
count, the softmax temperature and the new tagger's restoration, are instead
given the full interpretable span, from one qualitative extreme to the other
(always chasing the slowest runner through to chasing at random; a tenth of the
arousal gap restored through to nearly all of it), and the justification is that
one-parameter sweeps show the tag counts are insensitive to them across that
whole span, so the choice cannot bias the result. Either way the bounds are not
free guesses: they mark where behaviour stops changing, or are shown not to
matter, and because the contrast holds across the whole plausible region the
precise cut-offs do not affect the conclusion. 

---

## 7. Submodels

This section describes each process referenced in §3. Parameter names are given in
upper case.

### 7.1 Arousal decay

Each timestep, every **runner** (the tagger is exempt) loses arousal:

```
current_arousal_i  <-  max(0,  current_arousal_i  -  AROUSAL_DECAY_RATE * DELTA_TIME)
```

Arousal is bounded below at 0. `AROUSAL_DECAY_RATE` is a free parameter
expressed as arousal units lost per second. One boundary detail: on a catch
tick, roles are reassigned before decay is applied (§3.2), so the incoming
tagger skips that tick's decay and the outgoing tagger receives one tick of
decay despite having been tagger for the whole tick — immediately followed by
the restoration of §7.7, which dwarfs it.

### 7.2 Speed update

Each timestep, every agent recomputes its effective speed from its role and
arousal.

**Tagger:** `current_speed_i = max_speed_i`. The tagger always runs at full
intrinsic speed, regardless of arousal or condition.

**Runner:** speed is a weighted function of arousal,

```
current_speed_i  =  max_speed_i * ( 1  -  w * ( 1 - current_arousal_i ) )
```

where `w = condition_weight ∈ [0, 1]` determines how strongly the arousal
signal modulates speed:

- `w = 0` — arousal is ignored; the runner
  always moves at max speed. *Compete mode (WIN): the agent ignores its
  internal state and pursues the extrinsic goal.*
- `w = 1` — arousal fully scales speed,
  `current_speed_i = max_speed_i * current_arousal_i`. *Play mode (FUN): the
  agent's internal state drives behaviour, the formal analogue of
  self-handicapping in the verbal theory.*
- `0 < w < 1` — partial modulation.

The two condition weights, fixed at one in FUN and zero in WIN, are the sole
mechanism by which the conditions differ. Because `w` and `current_arousal_i` are both bounded in [0, 1], the
scaling factor is always in [0, 1] and `current_speed_i` is always
non-negative.

### 7.3 Target selection

When the tagger has no current target, it selects one. First the **eligible
runners** are determined: every agent that is not the tagger, not the
previous-tagger-immune agent, and not the abandoned-runner-immune agent. Then a
target is drawn at random from the eligible runners with probabilities given by a
softmax over the *negative* of their current speeds:

```
p_j  ∝  exp( - current_speed_j  /  SOFTMAX_TEMPERATURE )
```

Lower-speed runners therefore receive higher selection probability — the tagger
prefers slower prey. `SOFTMAX_TEMPERATURE` controls determinism: as it approaches 0
the tagger almost always picks the slowest eligible runner; as it grows large the
choice becomes uniform. (The softmax is computed with a max-subtraction step for
numerical stability.)

### 7.4 Chase dynamics

A chase is represented by the scalar `chase_distance`. When a target is first
selected, `chase_distance` is set to `STARTING_DISTANCE`. Each subsequent timestep:

```
closing_rate    =  tagger_speed  -  target_speed
noise           ~  Normal( 0,  PURSUIT_NOISE )
chase_distance  <-  chase_distance  -  closing_rate * DELTA_TIME  +  noise
```

`tagger_speed` is the tagger's current speed, which for a tagger is always its
maximum speed. The gap shrinks when the tagger is faster than the target and grows
when the target is faster; the Gaussian `noise` term, drawn fresh each step in
metres, represents agility, direction changes, terrain, and tactical variation
attributed to the interaction rather than to either individual. The noise is
per-tick, not scaled by the square root of the timestep (see §4.5). A negative
`chase_distance` is physically interpretable as the tagger having slightly
overshot. The `chase_timer` is advanced by one `DELTA_TIME` each step the chase
continues.

### 7.5 Catch and role switch

A **catch** occurs on any step where `chase_distance` falls to or below
`CATCH_DISTANCE`. On a catch:

1. The current tagger's id is stored in `immune_previous_tagger_id`, so that agent
   cannot be targeted for the entire duration of the new tagger's turn.
2. `immune_abandoned_id` is cleared — the new tagger begins with no abandoned-runner
   immunity.
3. The caught runner becomes the new tagger; all other agents become runners.
4. The chase state (`current_target`, `chase_distance`, `chase_timer`) is reset, so
   the new tagger selects a fresh target on the next step.
5. An event row is logged with the new tagger's `"<group>_<player>"` id and the
   timestamp of the catch.

Arousal restoration (Submodel §7.7) is applied later in the same step, after the
per-step arousal decay.

### 7.6 Abandonment

A chase is **abandoned** on any step where it is not a catch and either
`chase_distance` reaches or exceeds `GIVE_UP_DISTANCE` (the runner has pulled too far
away) or `chase_timer` reaches or exceeds `GIVE_UP_TIME` (the tagger has spent too
long on this target). On abandonment, the abandoned runner's id is stored in
`immune_abandoned_id` — replacing whoever was previously in that slot, which makes
the previous occupant eligible again — and the chase state is cleared so a new target
is selected next step. Abandonment does not change roles and is not logged as an
event.

### 7.7 Arousal restoration

When a catch occurs, after the per-step arousal decay (§7.1) and before speed is
recomputed (§7.2), every agent's arousal is restored toward the ceiling of 1.0.
Note the deliberate asymmetry with §7.1: decay is a linear per-tick seep,
restoration an event-triggered proportional jump toward the ceiling:

```
current_arousal_i  <-  1  -  ( 1 - current_arousal_i ) * exp( - restoration_param )
```

The further an agent is from full arousal, the larger its absolute gain — a
depleted agent recovers more than a fresh one — and arousal can never exceed 1.0.
The restoration parameter depends on the agent's new role:

- the **new tagger** receives `AROUSAL_RESTORATION_TAGGER` (a stronger
  restoration, representing the engagement spike of becoming the active chaser);
- **every other agent** receives `AROUSAL_RESTORATION_SWITCH` (a weaker
  restoration, representing the shared arousal of a tag event for bystanders).

Both restorations are swept, and the search keeps
`AROUSAL_RESTORATION_TAGGER > AROUSAL_RESTORATION_SWITCH` (§1.3). The new
tagger's restoration reaches behaviour only through the arousal it carries into
its next spell as a runner, since while it is the tagger it runs at full speed
regardless of arousal, but it is swept rather than fixed.

### 7.8 Game termination

A game runs for a fixed number of steps, `GAME_DURATION / DELTA_TIME`. The game ends
when `current_time` reaches `GAME_DURATION`; the model simply stops. **No
end-of-game row is logged.** The event log for a game therefore consists of the
`timestamp = 0` initial-tagger marker followed by one row per catch — and nothing
else. This matches the real data-collection tool, which likewise logs the initial
tagger and each tag, but logs nothing when the game's time expires.

---

## Appendix A — Parameter summary

| Parameter | Role | Status |
|---|---|---|
| `N_PLAYERS` | Players per group | Fixed by experiment design (5) |
| `GAME_DURATION` | Seconds per game | Fixed by experiment design (120 s) |
| `DELTA_TIME` | Timestep length (s) | Fixed, modelling choice (0.5 s) |
| `MU_MAX_SPEED` | Mean intrinsic speed (m/s) | Fixed, from data (6.0) |
| `SIGMA_MAX_SPEED` | SD of intrinsic speed (m/s) | Fixed, from data (0.5) |
| `CONDITION_WEIGHT_FUN` | Weight on arousal in FUN | Fixed by theory (1.0) |
| `CONDITION_WEIGHT_WIN` | Weight on arousal in WIN | Fixed by theory (0.0) |
| `MASTER_SEED` | Top-level RNG seed | Fixed |
| `GIVE_UP_DISTANCE` | Gap at or above which the tagger abandons (m) | **Swept** [3, 12] |
| `GIVE_UP_TIME` | Time before the tagger abandons a target (s) | **Swept** [3, 10] |
| `CATCH_DISTANCE` | Gap at or below which a catch occurs (m) | **Swept** [0.5, 2] |
| `STARTING_DISTANCE` | Initial tagger–target gap (m) | **Swept** [1.5, 5] |
| `AROUSAL_RESTORATION_TAGGER` | Restoration strength for the new tagger | **Swept** [0.1, 3] |
| `AROUSAL_RESTORATION_SWITCH` | Restoration strength for the other players | **Swept** [0.1, 3] |
| `AROUSAL_DECAY_RATE` | Arousal lost per second by a runner | **Swept** [0.005, 0.08] |
| `SOFTMAX_TEMPERATURE` | Randomness of target selection | **Swept** [0.1, 4] |
| `PURSUIT_NOISE` | SD of per-step chase-distance noise (m) | **Swept** [0, 1.5] |

*The nine swept parameters are sampled uniformly within their ranges in
`tag_abm_main_v2.ipynb`, subject to two redraw restrictions:
`CATCH_DISTANCE < STARTING_DISTANCE < GIVE_UP_DISTANCE` and
`AROUSAL_RESTORATION_TAGGER > AROUSAL_RESTORATION_SWITCH` (§1.3). Five hundred
sets are drawn, each evaluated on 100 paired games, and the sets whose average
game stays under the 60-tag ceiling form the reported region.*

---

## Appendix B — Key references

Andersen, M. M., Kiverstein, J., Miller, M., & Roepstorff, A. (2023). Play in
predictive minds: A cognitive theory of play. *Psychological Review*, 130(2),
462–479. — *The verbal predictive-processing theory of play that this model
formalises.*

Smaldino, P. E. (2017). Models are stupid, and we need more of them. In R. R.
Vallacher, S. J. Read, & A. Nowak (Eds.), *Computational Social Psychology*.
Routledge. — *The methodological framing for why a deliberately
oversimplified formal model is worth building from a verbal theory.*

Vandoni, M., Gatti, A., Carnevale Pellino, V., Cavallo, C., Pirazzi, A.,
Giuriato, M., & Lovecchio, N. (2024). Temporal trends of linear speed and
change of direction performance in Italian children. *Journal of Human
Kinetics*, 95, 215–226. — *The empirical anchor for the `MU_MAX_SPEED` and
`SIGMA_MAX_SPEED` speed distribution in §6 — 30 m sprint times for a large
sample of 11–13 year olds, drawing on the 30 m protocol of Lovecchio et al.
(2020).*

Grimm, V., Railsback, S. F., Vincenot, C. E., Berger, U., Gallagher, C.,
DeAngelis, D. L., et al. (2020). The ODD protocol for describing agent-based
and other simulation models: A second update to improve clarity, replication,
and structural realism. *Journal of Artificial Societies and Social
Simulation*, 23(2), 7. — *The ODD protocol followed in this document.*
