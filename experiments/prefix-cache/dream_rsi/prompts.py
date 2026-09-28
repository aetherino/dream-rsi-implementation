import json
import hashlib
import ast
from .scoring import TOTAL_EXTRA, objective_description
from .programs import POLICY_FEATURE_SETS

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
                               "saved_prompt_tokens", "score_contribution", "evicted_blocks", "policy_us_per_request", "workloads") if k in r}
            for r in feedback.get("runs", [])]}}


def discovery_prompt_v1(parent, visible, suite_summary, prior_rollouts=(), scoring=TOTAL_EXTRA):
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


def controller_prompt_v1(incumbent, feedback, history_summaries, settings, revision_log=(), scoring=TOTAL_EXTRA):
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


PROMPT_VERSIONS = {"v1", "v2"}
LANGUAGE_V2 = """The expression language accepts finite numeric constants, + - * /, comparisons,
and/or/not, x if condition else y, min(a,b), max(a,b), abs(x), log1p(x), sqrt(x).
No imports, attribute access, indexing, powers, extra functions, or arbitrary Python.
LIMITS ARE MAXIMUMS, NOT TARGETS: at most 1000 characters, 96 AST nodes, and nesting
depth 12 per expression. A short, shallow expression is valid; never pad it to meet a limit.
Every numeric literal must have magnitude <= 1000000; all evaluated intermediate
values must be finite with magnitude <= 1000000000000. Avoid division by zero.
Names must be 1-80 characters. Use a brief rationale of 1-2 sentences, at most 400 characters.
Before returning, check allowed names, JSON keys, arithmetic, constants, and expression size.
Return one JSON object only. Do not add markdown, validation reports, confidence fields,
parent IDs, or any other keys. Explanations are hypotheses; measured feedback is authoritative.
"""


def expression_key(expression):
    """Ignore formatting differences; this is not algebraic equivalence checking."""
    try:
        normalized = ast.dump(ast.parse(expression, mode="eval"), include_attributes=False)
    except (SyntaxError, ValueError, RecursionError):
        normalized = expression
    return hashlib.sha256(normalized.encode()).hexdigest()[:20]


def tried_memory(histories, tree, max_bytes=18000):
    """Training-only attempt ledger, bounded explicitly rather than silently truncated."""
    grouped = {}
    attempts = 0
    for cycle, history in enumerate([*histories, tree], 1):
        for node in history:
            if node["id"] == 0:
                continue
            candidate = node.get("candidate")
            expression = candidate.get("retention_score") if isinstance(candidate, dict) else None
            if not isinstance(expression, str):
                continue
            attempts += 1
            key = expression_key(expression)
            if key not in grouped:
                grouped[key] = {"key": key, "retention_score": expression, "attempts": 0, "best_valid_score": None}
            row = grouped[key]
            row["attempts"] += 1
            row.update(last_cycle=cycle, last_node=node["id"], last_error=node["feedback"].get("error"))
            if node["valid"]:
                row["best_valid_score"] = max(row["best_valid_score"] if row["best_valid_score"] is not None else -1e6, node["score"])
    rows = list(grouped.values())
    # Favor useful measured policies and recent attempts if a very long run needs compaction.
    ranked = sorted(rows, key=lambda r: (r["best_valid_score"] if r["best_valid_score"] is not None else -1e6,
                                         r["last_cycle"], r["last_node"]), reverse=True)
    included = []
    for row in ranked:
        if len(json.dumps(included + [row], ensure_ascii=False).encode()) <= max_bytes:
            included.append(row)
    return {"attempts_with_expressions": attempts, "unique_expression_count": len(rows),
            "included_count": len(included), "omitted_count": len(rows) - len(included), "expressions": included,
            "note": "Only training observations. Same AST ignores whitespace, not algebraic equivalence. Omitted formulas are not available in this prompt."}


def compact_node_v2(node):
    compact = compact_node(node)
    if isinstance(compact.get("candidate"), dict):
        compact["candidate"] = {k: v for k, v in compact["candidate"].items() if k != "rationale"}
    compact["feedback"].update({k: node["feedback"][k] for k in (
        "scoring", "total_extra_computed_tokens", "total_lru_extra_computed_tokens",
        "saved_prompt_tokens", "score_denominator_tokens") if k in node["feedback"]})
    return compact


def _discovery_prompt(parent, visible, suite_summary, prior_rollouts=(), scoring=TOTAL_EXTRA,
                     *, version="v2", best=None, memory=None, policy_features="block-v1"):
    if policy_features not in POLICY_FEATURE_SETS:
        raise ValueError("Unknown policy feature set")
    feature_help = ""
    if policy_features == "task-v1":
        if version != "v2":
            raise ValueError("Task features require prompt v2")
        feature_help = """
Additional allowed runtime names: task_chat, task_qa, task_unknown, turn_index.
The three task flags are booleans for the declared serving category of the MOST RECENT
request touching that block. They are not dataset IDs. Shared prefixes take the latest
toucher category. turn_index is that request's zero-based current turn (QA is always 0).
Chat includes conversations that may never return; no continuation or final-turn flag is supplied.
No raw text, token IDs, session IDs, document IDs, future arrivals or future hits are exposed.
Use only observable metadata; task labels do not guarantee future reuse.
"""
    if version == "v1":
        return discovery_prompt_v1(parent, visible, suite_summary, prior_rollouts, scoring)
    if version != "v2":
        raise ValueError("Unknown prompt version")
    return """Propose one improved prefix-cache eviction policy for the fixed CPU simulator.
Output exactly these keys: {"name":"...", "retention_score":"expression", "rationale":"..."}.
At each eviction, the eligible entry with the LOWEST score is removed; larger scores protect entries.
Ties use least recent access order. Only unpinned resident LEAVES are eligible; every block has equal size.
Example: in a resident chain A -> B -> C, only C is a leaf. Evicting C leaves A and B cached.
A later request can reuse A and B before recomputing C and any uncached requested tail.
An eligible leaf has no resident descendants to evict. Depth is metadata, not a direct multiplier
of recomputation cost. Ancestors may become eligible later after children are removed.
Allowed runtime names: now, depth, inserted_at, last_access, frequency, insertion_order, access_order.
Times are virtual request arrival times. Frequency counts accesses since insertion, including insertion.
Depth is prefix length in blocks; insertion_order/access_order are monotonic event counts.
Adding a common term such as now alone to every entry's score does not change their ranking.
All eligible entries are scored, so expression cost per entry contributes to total CPU policy time.
The CPU-time feasibility limit is in the suite. GPU latency is not measured.
Improve the selected parent, using the best-ever policy and measured outcomes as evidence.
Do not resubmit an expression in tried_expressions without a specific reason in the rationale.
Prefer a different hypothesis or a focused parameter change over renaming a previous formula.
The authoritative aggregate feedback supplies saved tokens, the LRU denominator, and score; do not invent totals.
Distinguish expected benefits from observed benefits. No future request information is available at runtime.
""" + feature_help + objective_description(scoring) + "\n" + LANGUAGE_V2 + "\n" + json.dumps({
        "prompt_version": "v2", "policy_features": policy_features, "selected_parent": compact_node_v2(parent),
        "best_ever_training_policy": compact_node_v2(best) if best else None,
        "recent_observations": [compact_node_v2(n) for n in visible[-12:]],
        "prior_rollout_best_observations": [compact_node_v2(n) for n in prior_rollouts],
        "tried_expressions": memory, "suite": suite_summary})



def discovery_prompt(parent, visible, suite_summary, prior_rollouts=(), scoring=TOTAL_EXTRA,
                     *, version="v2", best=None, memory=None, policy_features="block-v1"):
    """Bound serialized input, retaining parent, best, suite and attempted-formula ledger."""
    recent, prior = list(visible[-12:]), list(prior_rollouts)
    while True:
        prompt = _discovery_prompt(parent, recent, suite_summary, prior, scoring, version=version,
                                  best=best, memory=memory, policy_features=policy_features)
        messages = [{"role": "user", "content": prompt}]
        # Leave room for backend system text/envelope and the 4096-byte allowance.
        if len(json.dumps(messages, ensure_ascii=False).encode()) <= 54000:
            return prompt
        if recent:
            recent.pop(0)
        elif prior:
            prior.pop(0)
        else:
            raise ValueError("Selected parent, best, suite and memory exceed prompt allowance")


def controller_prompt(incumbent, feedback, history_summaries, settings, revision_log=(), scoring=TOTAL_EXTRA,
                      *, version="v2", online_rounds=None):
    if version == "v1":
        return controller_prompt_v1(incumbent, feedback, history_summaries, settings, revision_log, scoring)
    if version != "v2" or type(online_rounds) is not int or online_rounds < 1:
        raise ValueError("Prompt v2 requires the actual online round limit")
    return """Revise the exploration controller, not the cache policy.
Output exactly these keys: {"name":"...", "root_if":"expression", "root_priority":"expression",
"leaf_if":"expression", "leaf_priority":"expression", "rationale":"..."}.
Each online cycle begins at LRU. A round selects at most W DISTINCT eligible nodes from root plus observed leaves.
Selecting root opens ONE new branch. Selecting a leaf refines it ONCE; neither node can fill multiple slots.
First evaluate root_if/leaf_if. Rank eligible nodes by DESCENDING numeric priority, selecting the top W.
Example with W=2: root priority 3, leaf A priority 3.2, leaf B priority 2.9 selects leaf A and root.
The number of selected nodes can be less than W. No eligible actions means stop the cycle.
Check that priorities and conditions actually implement the behavior described in your rationale.
Globals: round_index (starts at 1 each cycle), attempts_seen, branches_opened, best_score.
Leaf expressions also receive score, gain (versus parent), depth, stagnation (consecutive nonimprovements),
failure_streak, valid (boolean), age_rounds. No future scores, node IDs, or hidden histories at runtime.
deployment.online_rounds_per_cycle is the LIVE limit. replay_settings.max_rounds is the OFFLINE limit;
these can differ. A strategy depending on later rounds will not get those rounds in live discovery.
Replay value = best observed task score - beta_calls*N + beta_parallel*N/max(1,rounds), averaged across histories.
N includes failed recorded attempts. Initial LRU with score 0 is always available.
Replay only reveals recorded outcomes. A missing continuation is unobserved, not a known bad result;
selecting it uses a round slot, marks it exhausted, and adds no generation-call charge in replay.
Prefer rules likely to transfer to new discovery, not rules that merely avoid known bad recorded nodes.
Historical replay improvement is not proof of live-search improvement. A proposal is accepted only if its
replay value strictly exceeds the incumbent on the same history pool. Do not invent unseen outcomes.
""" + objective_description(scoring) + "\n" + LANGUAGE_V2 + "\n" + json.dumps({
        "prompt_version": "v2", "deployment": {"online_rounds_per_cycle": online_rounds,
            "workers": settings["workers"], "max_depth": settings["max_depth"]},
        "incumbent": incumbent, "incumbent_replay": feedback, "historical_trees": history_summaries,
        "replay_settings": settings, "previous_revisions": [{"controller": r["controller"],
            "accepted": r["accepted"], "evaluation": r["evaluation"]} for r in revision_log]})
