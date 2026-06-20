"""
tag_abm.py
==========
Mesa-based agent-based model of agents playing tag.

This model is a minimal formalisation of Andersen et al.'s (2023) predictive
processing account of play. Each agent carries a private *arousal* state in
[0, 1]. Arousal decays while the agent is a runner, is restored on tag
events, and modulates running speed via a per-condition *condition weight*.
The two conditions (FUN, WIN) differ only in this weight: in FUN the weight
on arousal is high, so a depleted arousal signal pulls speed down; in WIN
the weight is low, so the runner ignores its internal state and simply runs
at top speed.

This is a Smaldino-style "stupid model" — its purpose is to make the verbal
theory precise enough to scrutinise, not to fit data. See the calibration
notebook for the explicit list of structural commitments.

Usage
-----
from tag_abm import run_experiment
events_df, agent_df = run_experiment(params)

Output DataFrames
-----------------
events_df : one row per tagging event
    columns: group, condition, runner_tagged_id, timestamp

    The first row of each game (timestamp = 0) records the randomly
    assigned initial tagger. It is a marker, not a real tag, and is
    dropped before analysis. Every later row is a real tag. No
    end-of-game boundary row is logged — the game simply stops when the
    clock reaches GAME_DURATION, mirroring the real data-collection
    tool, which logs nothing at game end.

    runner_tagged_id is a cross-group-unique string id of the form
    "<group>_<player>" (e.g. group 0 player 5 -> "0_5"). This guarantees
    that player 1 of group 0 is never confused with player 1 of group 1.
    Groups are numbered (0, 1, 2, ...) so this simulated id is always
    distinguishable from real experimental data, where groups are
    lettered.

agent_df : one row per agent per timestep
    columns: group, condition, step, time, player,
             current_arousal_i, current_speed_i, max_speed_i, is_tagger
"""

import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.time import BaseScheduler


# ── Helper functions ───────────────────────────────────────────────────────

def softmax_neg(values, softmax_temperature):
    """Softmax over negative values: lower values get higher probability.

    Used by tagger to favour targeting slower runners.
    softmax_temperature -> 0   : always targets slowest (argmin)
    softmax_temperature -> inf : uniform random selection
    """
    v = np.array(values, dtype=float)

    # Negate and scale by temperature: lower speeds become larger values
    v_neg = -v / softmax_temperature

    # Subtract max for numerical stability (prevents exp overflow)
    max_value = v_neg.max()
    v_neg     = v_neg - max_value

    e = np.exp(v_neg)
    return e / e.sum()


# ── Agent ──────────────────────────────────────────────────────────────────

class PlayerAgent(Agent):
    """One player in the tag game.

    Inherits from Mesa's Agent class, which provides unique_id and self.model.

    Parameters
    ----------
    unique_id : int
        Player index [1, N]. Required by Mesa to tell agents apart.
    model : TagModel
        The game this agent belongs to. Accessible via self.model.
    max_speed_i : float
        Maximum running speed in m/s. Drawn once per group, never changes.
    """

    def __init__(self, unique_id, model, max_speed_i):
        super().__init__(unique_id, model)

        # Maximum (intrinsic) speed in m/s — fixed trait, never changes
        self.max_speed_i = max_speed_i

        # Effective speed this tick — derived from max_speed_i and arousal
        self.current_speed_i = max_speed_i

        # Arousal level in [0, 1] — starts fully excited, decays while a runner.
        self.current_arousal_i = 1.0

        # Role flag — True if this agent is currently the tagger
        self.is_tagger = False

    # ── Per-step updates ──────────────────────────────────────────────────

    def update_arousal(self):
        """Arousal decays each tick for runners. Tagger is always excluded.
        Arousal is bounded in [0, 1]. The floor is enforced here;
        the ceiling is maintained by the restoration formula which
        decays toward 1.0 and can never exceed it.
        """
        if not self.is_tagger:
            new_arousal = self.current_arousal_i - self.model.arousal_decay_rate * self.model.delta_time
            self.current_arousal_i = max(new_arousal, 0.0)

    def update_speed(self):
        """Speed is determined by role, arousal, and the condition weight.

        Tagger:
            Always runs at full max speed, regardless of arousal level
            or condition. The tagger is always fully engaged.

        Runner:
            Speed is a weighted function of arousal:

                current_speed = max_speed * (1 - w * (1 - current_arousal))

            where w = self.model.condition_weight in [0, 1] sets how
            strongly the arousal signal modulates speed.

            w = 0  : arousal is ignored — the runner always moves at max
                     speed (compete mode, WIN: goal-directed, the agent
                     ignores its internal state).
            w = 1  : arousal scales speed fully;
                     current_speed = max_speed * current_arousal
                     (play mode, FUN: the internal state drives
                     behaviour).
            0<w<1  : partial modulation.

            CONDITION_WEIGHT_FUN is high; CONDITION_WEIGHT_WIN is low.
            This is the only mechanism by which condition affects behaviour
            in the model — every other process is condition-invariant.
        """
        if self.is_tagger:
            self.current_speed_i = self.max_speed_i
        else:
            weight = self.model.condition_weight
            speed_factor = 1.0 - weight * (1.0 - self.current_arousal_i)
            self.current_speed_i = self.max_speed_i * speed_factor

    def apply_arousal_restoration(self, is_new_tagger):
        """Arousal restoration applied to this agent when a tag event occurs.

        Uses a proportional jump toward the ceiling:

            arousal_after = 1 - (1 - arousal_before) * exp(-restoration_param)

        Note the deliberate asymmetry with arousal decay: decay (update_arousal)
        is a linear per-tick seep, while restoration is an event-triggered,
        proportional jump toward 1.0. The further from 1.0 an agent is, the
        larger their absolute restoration — a depleted agent recovers more
        than a fresh one — and arousal can never exceed 1.0.

        New tagger receives a stronger restoration (AROUSAL_RESTORATION_TAGGER),
        reflecting the engagement spike from becoming the active chaser.

        All other players receive a smaller restoration (AROUSAL_RESTORATION_SWITCH),
        reflecting the shared arousal of a tag event for bystanders.
        """
        if is_new_tagger:
            restoration_param = self.model.arousal_restoration_tagger
        else:
            restoration_param = self.model.arousal_restoration_switch

        gap_to_ceiling            = 1.0 - self.current_arousal_i
        restored_gap              = gap_to_ceiling * np.exp(-restoration_param)
        self.current_arousal_i = 1.0 - restored_gap

    # Note: this agent has no step() method. The order in which arousal
    # decay, tag restoration, and speed are applied matters (see TagModel.step),
    # so the model calls update_arousal / apply_arousal_restoration /
    # update_speed explicitly rather than delegating to a scheduler.


# ── Model ──────────────────────────────────────────────────────────────────

class TagModel(Model):
    """One game of tag.

    Inherits from Mesa's Model class, which provides basic infrastructure.

    Parameters
    ----------
    group_id : int
        Which group is playing. Written into every output row.
    condition : str
        One of 'WIN' or 'FUN'. Selects which condition weight is used
        (a high weight in FUN, a low weight in WIN). This is the only
        mechanism by which condition affects behaviour.
    speeds : array of float
        Max speed for each player. Same across conditions within a group.
    params : dict
        All model hyperparameters. See run_experiment() for full list of keys.
    seed : int
        RNG seed for this specific game, ensuring reproducibility.
    """

    def __init__(self, group_id, condition, speeds, params, seed):
        super().__init__()

        # ── Store parameters ──────────────────────────────────────────────

        self.group_id                   = group_id
        self.condition                  = condition
        self.delta_time                 = params["DELTA_TIME"]
        self.arousal_decay_rate         = params["AROUSAL_DECAY_RATE"]
        self.arousal_restoration_tagger = params["AROUSAL_RESTORATION_TAGGER"]
        self.arousal_restoration_switch = params["AROUSAL_RESTORATION_SWITCH"]
        self.softmax_temperature        = params["SOFTMAX_TEMPERATURE"]
        self.starting_distance          = params["STARTING_DISTANCE"]
        self.catch_distance             = params["CATCH_DISTANCE"]
        self.give_up_distance           = params["GIVE_UP_DISTANCE"]
        self.give_up_time               = params["GIVE_UP_TIME"]
        self.pursuit_noise              = params["PURSUIT_NOISE"]
        self.game_duration              = params["GAME_DURATION"]

        # Condition weight — how strongly the arousal signal is allowed
        # to modulate runner speed. High in FUN (play mode: internal
        # state matters), low in WIN (compete mode: the extrinsic goal
        # dominates). See PlayerAgent.update_speed.
        if condition == "WIN":
            self.condition_weight = params["CONDITION_WEIGHT_WIN"]
        else:
            self.condition_weight = params["CONDITION_WEIGHT_FUN"]

        # ── Initialise state ──────────────────────────────────────────────

        self.rng          = np.random.default_rng(seed)
        self.current_time = 0.0
        self.current_step = 0
        self.event_rows   = []
        self.game_over    = False

        # Immunity slot 1: the previous tagger can never be targeted.
        # Set on every tag event. Persists until the next tag event.
        self.immune_previous_tagger_id = None

        # Immunity slot 2: the most recently abandoned runner cannot be targeted.
        # Set on every abandon. Replaced by the next abandon or cleared on a tag event.
        self.immune_abandoned_id = None

        # Chase state
        self.current_target  = None   # runner agent currently being chased (or None)
        self.chase_distance  = 0.0    # current distance between tagger and target
        self.chase_timer     = 0.0    # time elapsed chasing current target

        # ── Create agents ─────────────────────────────────────────────────

        self.schedule = BaseScheduler(self)

        for i in range(len(speeds)):
            # unique_id starts at 1 for readability in output
            agent = PlayerAgent(i + 1, self, float(speeds[i]))
            self.schedule.add(agent)

        # ── Agent-state log ───────────────────────────────────────────────
        # One record per agent per timestep is appended here. Collection is
        # explicit (see _collect_agents) rather than via Mesa's DataCollector,
        # because the model drives agent updates directly instead of through
        # schedule.step() — see TagModel.step for why.

        self.agent_records = []

        # ── Assign first tagger ───────────────────────────────────────────

        # Sample from [1, N+1) because unique_ids start at 1
        first_tagger_id = int(self.rng.integers(1, len(speeds) + 1))
        self._set_tagger(first_tagger_id)

        # Log the initial assignment at t=0. This is a marker, not a true
        # tag event, and is dropped before analysis.
        self.event_rows.append({
            "group":            group_id,
            "condition":        condition,
            "runner_tagged_id": self._player_label(first_tagger_id),
            "timestamp":        0.0,
        })

        # Collect initial agent states before any steps run
        self._collect_agents()

    # ── Private helpers ───────────────────────────────────────────────────

    def _set_tagger(self, agent_id):
        """Set one agent as tagger and all others as runners."""
        for agent in self.schedule.agents:
            if agent.unique_id == agent_id:
                agent.is_tagger = True
            else:
                agent.is_tagger = False

    def _get_tagger(self):
        """Find and return the current tagger agent."""
        for agent in self.schedule.agents:
            if agent.is_tagger:
                return agent
        raise RuntimeError("No tagger found.")

    def _player_label(self, player_number):
        """Build the cross-group-unique player id for an output row.

        The id has the form "<group>_<player>", e.g. group 0 player 5
        becomes "0_5". This keeps player 1 of group 0 distinct from
        player 1 of group 1 ("1_1") in the events DataFrame. Groups are
        numbered, so this simulated id never collides with real
        experimental data, where groups are lettered.
        """
        return f"{self.group_id}_{player_number}"

    def _collect_agents(self):
        """Append one record per agent capturing current state this step."""
        for agent in self.schedule.agents:
            self.agent_records.append({
                "step":                 self.current_step,
                "time":                 self.current_time,
                "player":               agent.unique_id,
                "current_arousal_i": agent.current_arousal_i,
                "current_speed_i":      agent.current_speed_i,
                "max_speed_i":          agent.max_speed_i,
                "is_tagger":            agent.is_tagger,
            })

    def _get_eligible_targets(self):
        """Return all runners who are currently eligible to be targeted.

        Two immunity slots exclude runners from selection:

        immune_previous_tagger_id:
            The player who was tagger immediately before the current tagger.
            This immunity persists for the entire duration of the current
            tagger's turn — it never clears until the next tag event.

        immune_abandoned_id:
            The runner most recently abandoned by the tagger this turn.
            This immunity clears when the tagger abandons again (replaced
            by the newly abandoned runner), meaning the previously immune
            runner becomes eligible again.

        A runner must be neither the current tagger nor immune under either
        slot to be eligible.
        """
        eligible = []
        for agent in self.schedule.agents:
            if agent.is_tagger:
                continue
            if agent.unique_id == self.immune_previous_tagger_id:
                continue
            if agent.unique_id == self.immune_abandoned_id:
                continue
            eligible.append(agent)
        return eligible

    def _select_target(self):
        """Tagger selects a target via softmax over current runner speeds.

        Lower speed -> higher probability of being targeted.
        softmax_temperature controls how deterministic the selection is:
            low temperature  -> tagger almost always picks the slowest runner
            high temperature -> tagger picks nearly at random
        """
        eligible   = self._get_eligible_targets()

        # With N players, the tagger and the two immunity slots exclude at
        # most three agents, leaving N - 3 eligible runners. The model
        # therefore assumes N_PLAYERS >= 4 (the design uses 5).
        assert eligible, (
            "No eligible targets: _select_target requires N_PLAYERS >= 4 "
            "(tagger + two immunity slots can exclude up to three agents)."
        )

        speed_list = []

        for runner in eligible:
            speed_list.append(runner.current_speed_i)

        probs = softmax_neg(speed_list, self.softmax_temperature)
        idx   = self.rng.choice(len(eligible), p=probs)
        return eligible[idx]

    def _apply_tag(self, new_tagger_agent):
        """Execute a role switch when a catch occurs.

        1. Set immune_previous_tagger_id to the current tagger (who is about
           to become a runner). They can never be targeted this turn.
        2. Clear immune_abandoned_id — the new tagger starts with a clean
           abandon slate. Only the previous-tagger immunity carries over.
        3. Reassign roles.
        4. Reset chase state for the incoming tagger.

        Note: arousal restorations are applied in step() after the
        per-step arousal decay, so all agents accumulate one final tick
        of decay before the restoration takes effect.
        """

        # The current tagger becomes permanently immune for this turn
        old_tagger                     = self._get_tagger()
        self.immune_previous_tagger_id = old_tagger.unique_id

        # New tagger starts with no abandoned-runner immunity
        self.immune_abandoned_id = None

        # Reassign roles
        self._set_tagger(new_tagger_agent.unique_id)

        # Reset chase state — new tagger starts fresh with no target
        self.current_target = None
        self.chase_distance = 0.0
        self.chase_timer    = 0.0

    def _abandon_chase(self):
        """Tagger gives up on current target and prepares to pick a new one.

        The abandoned runner becomes immune_abandoned_id, replacing whoever
        was previously in that slot (making the previous abandonee eligible
        again). The tagger will pick a new target on the next tick.
        """
        # The runner just abandoned becomes the new abandoned-immune agent
        self.immune_abandoned_id = self.current_target.unique_id

        # Clear chase state so a new target is selected next tick
        self.current_target = None
        self.chase_distance = 0.0
        self.chase_timer    = 0.0

    # ── Main step ─────────────────────────────────────────────────────────

    def step(self):
        """Advance the model by one time step of duration delta_time.

        Per-step sequence:
        1.  If no current target: select one from eligible runners
        2.  Update chase distance based on speed difference and noise
        3.  Advance chase timer
        4.  Evaluate exit conditions:
              distance <= catch_distance  -> catch: switch roles, log event
              distance >= give_up_distance -> abandon: set abandoned immunity,
                                             clear target, pick new next tick
              chase_timer >= give_up_time  -> abandon: same as above
        5.  Arousal update — time decay (every runner loses arousal)
        6.  Arousal update — tag restoration (only if a catch occurred:
              every agent's arousal is restored)
        7.  Speed update — every agent recomputes speed once, from the
              now-final arousal
        8.  Advance clocks and collect data
        9.  Check for game end

        Update ordering
        ---------------
        Arousal is fully resolved before speed is touched. Each agent's
        arousal changes first from the passage of time (step 5) and then,
        if a tag occurred, from the tag event (step 6); only then is speed
        computed (step 7), exactly once, from the final arousal. There is
        no second speed calculation: steps 5–7 each run once per tick.

        Chase distance mechanics
        ------------------------
        When a new target is selected, chase_distance is initialised to
        STARTING_DISTANCE — the physical gap the tagger must close.

        Each tick:
            chase_distance = chase_distance
                             - (tagger_speed - runner_speed) * dt
                             + noise

        noise ~ N(0, PURSUIT_NOISE^2), drawn fresh each tick in metres.
        It represents agility, direction changes, terrain, and tactical
        variation — not a property of either agent individually but of
        the interaction. The noise is per-tick, not scaled by sqrt(dt),
        so the dynamics are tied to the fixed DELTA_TIME of 0.5 s;
        changing DELTA_TIME would change the model, not just its
        resolution.

        A catch occurs when chase_distance <= CATCH_DISTANCE (tagger is
        within reach). Negative distance is physically interpretable as
        the tagger having overshot slightly due to noise.

        No end-of-game row is logged in step 9 — the game simply stops.
        """

        if self.game_over:
            return

        dt     = self.delta_time
        tagger = self._get_tagger()

        # ── 1. Select target if none exists ───────────────────────────────
        if self.current_target is None:
            self.current_target = self._select_target()
            self.chase_distance = self.starting_distance
            self.chase_timer    = 0.0

        target = self.current_target

        # ── 2. Update chase distance ──────────────────────────────────────
        # The tagger always moves at max speed (current_speed_i equals
        # max_speed_i for a tagger). The runner moves at current_speed_i,
        # which is scaled by arousal in proportion to the condition
        # weight (strongly in FUN, weakly in WIN).
        closing_rate        = tagger.current_speed_i - target.current_speed_i
        noise               = self.rng.normal(0.0, self.pursuit_noise)
        self.chase_distance = self.chase_distance - closing_rate * dt + noise

        # ── 3. Advance chase timer ────────────────────────────────────────
        self.chase_timer = self.chase_timer + dt

        # ── 4. Evaluate exit conditions ───────────────────────────────────

        # Track whether a catch happened this step — needed for step 6
        catch_occurred       = False
        new_tagger_this_step = None

        if self.chase_distance <= self.catch_distance:
            # Tagger is within catch range — tag occurs
            new_tagger_this_step = target
            catch_occurred       = True

            self._apply_tag(new_tagger_agent=target)
            self.event_rows.append({
                "group":            self.group_id,
                "condition":        self.condition,
                "runner_tagged_id": self._player_label(new_tagger_this_step.unique_id),
                "timestamp":        round(self.current_time + dt, 3),
            })

        elif self.chase_distance >= self.give_up_distance:
            # Runner has pulled too far away — abandon
            self._abandon_chase()

        elif self.chase_timer >= self.give_up_time:
            # Tagger has spent too long on this target — abandon
            self._abandon_chase()

        # ── 5. Arousal update — time decay ─────────────────────────────
        # Every runner loses arousal to the passage of time. On a catch
        # tick, roles are already reassigned above, so the new tagger
        # (is_tagger=True) skips decay; the demoted tagger now decays.
        for agent in self.schedule.agents:
            agent.update_arousal()

        # ── 6. Arousal update — tag restoration ────────────────────────
        # Only on a catch tick. Applied after the time decay of step 5, so
        # every agent decays one final tick before being restored.
        if catch_occurred:
            for agent in self.schedule.agents:
                is_new_tagger = (agent.unique_id == new_tagger_this_step.unique_id)
                agent.apply_arousal_restoration(is_new_tagger=is_new_tagger)

        # ── 7. Speed update — computed once, from the final arousal ────
        for agent in self.schedule.agents:
            agent.update_speed()

        # ── 8. Advance clocks and collect data ────────────────────────────
        self.current_step = self.current_step + 1
        self.current_time = round(self.current_time + dt, 3)
        self._collect_agents()

        # ── 9. Check for game end ─────────────────────────────────────────
        # The game simply stops; no end-of-game boundary row is logged.
        if self.current_time >= self.game_duration:
            self.game_over = True

    def run(self):
        """Run the full game from start to finish.

        Returns
        -------
        event_rows : list of dicts
            All logged rows: the initial-tagger marker (timestamp 0)
            followed by one row per tag. No end-of-game row is logged.
        """
        n_steps = int(self.game_duration / self.delta_time)

        for _ in range(n_steps):
            if self.game_over:
                break
            self.step()

        return self.event_rows


# ── Experiment runner ──────────────────────────────────────────────────────

def run_experiment(params):
    """Run the full experiment across all groups and conditions.

    Max speeds are drawn once per group and reused across all conditions,
    matching the within-subjects experimental design.

    Parameters
    ----------
    params : dict with keys:
        N_GROUPS                    : int   — number of groups
        N_PLAYERS                   : int   — players per group
        GAME_DURATION               : float — seconds per game
        CONDITIONS                  : list  — ['WIN', 'FUN']
        MU_MAX_SPEED                : float — mean max speed (m/s)
        SIGMA_MAX_SPEED             : float — SD of max speed
        AROUSAL_DECAY_RATE          : float — arousal lost per second while a runner
        AROUSAL_RESTORATION_TAGGER  : float — restoration parameter for new tagger
        AROUSAL_RESTORATION_SWITCH  : float — restoration parameter for all other players
        CONDITION_WEIGHT_FUN        : float — weight on arousal in FUN (high — internal state drives speed)
        CONDITION_WEIGHT_WIN        : float — weight on arousal in WIN (low — extrinsic goal dominates)
        DELTA_TIME                  : float — time step in seconds
        SOFTMAX_TEMPERATURE         : float — randomness of target selection
        STARTING_DISTANCE           : float — initial distance between tagger and target (m)
        CATCH_DISTANCE              : float — distance at which a catch occurs (m)
        GIVE_UP_DISTANCE            : float — distance at which tagger abandons (m)
        GIVE_UP_TIME                : float — time in seconds before tagger abandons
        PURSUIT_NOISE               : float — SD of positional noise per tick (m)
        MASTER_SEED                 : int   — top-level RNG seed

    Returns
    -------
    events_df : pd.DataFrame
        columns: group, condition, runner_tagged_id, timestamp
        runner_tagged_id is a "<group>_<player>" string id (e.g. "0_5").
        Per game: one t=0 initial-tagger marker row followed by one row
        per tag. No end-of-game boundary row. The t=0 row is a marker,
        not a real tag, and should be dropped before analysis.

    agent_df : pd.DataFrame
        columns: group, condition, step, time, player,
                 current_arousal_i, current_speed_i, max_speed_i, is_tagger
        One row per agent per timestep across all games.
    """

    rng        = np.random.default_rng(params["MASTER_SEED"])
    all_events = []
    all_agents = []

    for group in range(params["N_GROUPS"]):

        # Draw max speeds for this group from a Gaussian, clipped below
        # at 0.5 m/s (a hard clip, not a true truncation; at the
        # calibrated mu and sigma the clip has negligible probability mass)
        speeds = rng.normal(params["MU_MAX_SPEED"], params["SIGMA_MAX_SPEED"], params["N_PLAYERS"])
        speeds = np.clip(speeds, 0.5, None)

        for condition in params["CONDITIONS"]:

            seed  = int(rng.integers(0, 2**31))
            model = TagModel(group, condition, speeds, params, seed)
            model.run()

            all_events.extend(model.event_rows)

            # Agent-level data is collected explicitly by the model
            agent_data = pd.DataFrame(model.agent_records)
            agent_data["group"]     = group
            agent_data["condition"] = condition

            agent_data = agent_data[[
                "group", "condition", "step", "time",
                "player", "current_arousal_i", "current_speed_i",
                "max_speed_i", "is_tagger"
            ]]

            all_agents.append(agent_data)

        # Progress report after each group completes all conditions
        speed_strings = []
        for s in speeds:
            speed_strings.append(f"{s:.2f}")
        print(f"Group {group:2d}  |  speeds: {speed_strings}")

    events_df = pd.DataFrame(all_events)
    agent_df  = pd.concat(all_agents, ignore_index=True)

    return events_df, agent_df
