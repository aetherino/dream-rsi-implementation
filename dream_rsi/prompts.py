import json
from .scoring import TOTAL_EXTRA, objective_description

LANGUAGE = """Expressions support finite numbers, + - * /, comparisons, and/or/not, x if condition else y,
and min(a,b), max(a,b), abs(x), log1p(x), sqrt(x). No other syntax or functions.
At most 1000 characters, 96 AST nodes and nesting depth 12 per expression.
Constants have magnitude <= 1e6; all intermediate results must be finite and <= 1e12 in magnitude.
Avoid division by zero. Return only a JSON object, no markdown or commentary outside it.
Names <=80 characters and optional rationale <=4000 characters.
"""


def compact_node(node):
    feedback = node["feedback"]
    return {k: node[k] for k in ("id", "parent", "depth", "score", "valid", "candidate")} | {
        "feedback": {"error": feedback.get("error"), "runs": [
            {k: r[k] for k in ("name", "extra_computed_tokens", "lru_extra_computed_tokens",
                               "saved_prompt_tokens", "score_contribution", "evicted_blocks", "policy_us_per_request") if k in r}
            for r in feedback.get("runs", [])]}}


def discovery_prompt(parent, visible, suite_summary, prior_rollouts=(), scoring=TOTAL_EXTRA):
    return """Propose an improved prefix-cache eviction policy for the fixed CPU simulator.
The lowest retention_score among eligible entries is evicted. Ties use least recent access order.
Entries are equal-sized blocks, only unpinned leaves are eligible. Ancestors remain resident.
Available variables: now, depth, inserted_at, last_access, frequency, insertion_order, access_order.
Times are virtual request arrival times; frequency is access count since insertion; depth is prefix blocks.
Extra tokens are prompt computation beyond an unlimited-cache replay. Higher score is better; initial LRU=0.
Policies must also satisfy a CPU-time feasibility limit. Real GPU latency is not measured.
Output schema: {"name": "...", "retention_score": "expression", "rationale": "..."}.
Improve the selected parent, using only allowed entry metadata at runtime.
""" + objective_description(scoring) + "\n" + LANGUAGE + "\n" + json.dumps({"selected_parent": compact_node(parent),
        "recent_observations": [compact_node(n) for n in visible[-12:]],
        "prior_rollout_best_observations": [compact_node(n) for n in prior_rollouts], "suite": suite_summary})


def controller_prompt(incumbent, feedback, history_summaries, settings, revision_log=(), scoring=TOTAL_EXTRA):
    return """Revise the exploration controller, not the cache policy.
It chooses a batch of at most W distinct nodes from the root plus observed leaves each round.
Selecting root opens one new branch; selecting a leaf refines it once. Stop if no eligible actions.
The harness evaluates root_if/leaf_if then ranks descending root_priority/leaf_priority.
Output schema: {"name":"...", "root_if":"expression", "root_priority":"expression",
"leaf_if":"expression", "leaf_priority":"expression", "rationale":"..."}.
Globals available to all expressions: round_index (starts 1), attempts_seen, branches_opened, best_score.
Leaf expressions additionally get score, gain (vs parent), depth, stagnation (consecutive nonimprovements),
failure_streak, valid (boolean), age_rounds. No future scores, node IDs or hidden history at runtime.
Optimize mean replay value: best observed task score - beta_calls*N + beta_parallel*N/max(1,rounds).
N counts revealed recorded attempts including failures; initial LRU score 0 is always available.
Replay can only reveal already recorded outcomes; it cannot invent a continuation. Missing continuations
consume a round slot and become exhausted, with no new generation-call charge. The incumbent is retained
unless a revision strictly improves value on the identical history pool. Avoid overfitting these histories.
""" + objective_description(scoring) + "\n" + LANGUAGE + "\n" + json.dumps({"incumbent": incumbent, "incumbent_replay": feedback,
        "historical_trees": history_summaries, "settings": settings,
        "previous_revisions": [{"controller": r["controller"], "accepted": r["accepted"],
                                 "evaluation": r["evaluation"]} for r in revision_log]})
