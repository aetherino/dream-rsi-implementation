"""Executable Dream-RSI program prompts with complete saved development history.

Public interfaces mirror the expression harness where useful, but do not compact
or truncate history. Pass ``history`` to discovery_messages for all completed
prior cycles (it supersedes prior_rollouts), and ``visible`` for the entire current
revealed tree. Every supplied record is serialized unchanged, including source,
rationale, measured score, error and saved evaluation diagnostics. The baseline
should be supplied once by the caller; no best-only or recent-only ledger replaces
records. Bounds apply to the resulting messages through source_allowance.

controller_messages receives the latest sequential ``current_version``, its
feedback, the full completed history pool and every preceding revision. Candidate
controller execution receives only the current revealed prefix, never this author
prompt or hidden replay continuations.
"""
import json

from .code_transport import MAX_INPUT_ALLOWANCE, SYSTEM, source_allowance
from .scoring import TOTAL_EXTRA, objective_description

PUBLIC_PROGRAM = """Return exactly one JSON object with these three keys:
{"name":"short program name", "source":"complete self-contained Python source", "rationale":"reasoned hypothesis"}.
Name must have 1-80 characters, source at most 65536 UTF-8 bytes, rationale at most 4000.
The source must contain the requested class with a no-argument constructor. It is executable
Python, so you may use persistent instance state, loops, functions, branches, containers and
standard-library imports, including math, collections, heapq and statistics. No expression DSL.
The program runs in an isolated resource-limited child with no network or dataset access.
Do not read files, environment, raw request text, tokens, token IDs, dataset IDs, session IDs,
future arrivals or future outcomes. Opaque IDs are identity/selection handles only, never
predictive signals. Return no markdown, hidden reasoning fields, or other JSON keys.
Measured score and error feedback is authoritative; distinguish hypotheses from measurements.
"""

POLICY = """Propose one improved prefix-cache eviction program for the fixed CPU simulator.
Implement class CachePolicy with choose(self, eligible, now). eligible is the nonempty list
of unpinned resident LEAF entries. Return exactly one eligible entry.block_id to evict.
Every entry is an equal-size block; ancestors remain resident. A leaf has no resident descendants.
Evicting the tail of a chain permits its next ancestor to become eligible at a later eviction.
Use the current block metadata and your own state accumulated from observable events.
Allowed Entry attributes: block_id, depth, inserted_at, last_access, frequency,
insertion_order, access_order. Entry is immutable. block_id is an opaque identity handle;
its string content has no semantic meaning and must not be used as a prediction feature.
now is the current virtual request arrival time; inserted_at and last_access use that clock.
frequency counts accesses since insertion, including insertion; depth is prefix blocks.
insertion_order and access_order are monotonic observable event counters.
Optional event methods are observe_insert(self, entry), observe_hit(self, entry), and
observe_evict(self, entry). They take one entry, without a now argument. Events arrive in exact
observed order before the next choose call and are flushed after replay. Instance state
persists within a cache replay and resets for a fresh replay. A state table may use block_id
as a key to associate events with entries. No hidden information is supplied to these hooks.
CPU policy time is measured; GPU latency is not measured. Invalid execution is scored as failure.
Improve the selected parent using all completed proposals and current observations, including
failed attempts. Preserve useful hypotheses or revise them based on complete measured feedback.
"""

CONTROLLER = """Revise the exploration program sequentially from current_version below.
Implement class ExplorationController with a no-argument constructor and
select(self, observed_nodes, round_index, workers, max_depth). Return a JSON-serializable
list of at most workers DISTINCT integer node IDs, all legal before this round's batch starts.
observed_nodes contains root plus the currently revealed proposal prefix. Each node may have
id, parent, depth, round, score, valid, stagnation, failure_streak, candidate and feedback;
exhausted=True indicates a prior selection with no recorded continuation. Such a node remains
legal; selecting it again usually spends a replay round without revealing a new outcome.
Only root (id 0) and current leaves with depth < max_depth are legal actions. A current leaf
has no child in observed_nodes. Root is always legal regardless of max_depth. Root can be
selected at most once in a round, opening one new branch. Selecting a leaf refines it once.
Do not select nonexistent nodes, internal nonroot nodes, repeated IDs within a batch,
or nonroot nodes at max_depth.
Returning [] stops the episode. round_index starts at 1; the controller resets per episode.
You may inspect the full revealed trajectory and keep state based on past observed calls.
Node IDs identify actions and parent relationships; do not hard-code future outcomes or use
IDs as predictive features. Runtime cannot inspect author histories or unrevealed continuations.
The author sees all completed histories for development. Offline replay reveals outcomes only
after selection, in chronological rounds; it never invents missing continuations. A selected
missing continuation becomes exhausted and incurs a replay round but no generation-call charge.
The Section 3 Equation 1 objective is best revealed task score - beta_calls * N
+ beta_parallel * N / max(1, nonempty_rounds), averaged over the identical replay history pool.
N counts all revealed nonroot attempts, including failures. Baseline score 0 remains available.
beta_calls and beta_parallel are the explicit configured coefficients in settings/objective.
Live online-round limits can differ from offline replay limits; deployment is authoritative.
Each revision starts from the latest sequential current_version, including a worse-scoring
revision, retaining every earlier proposal, rationale, measured value, error and evaluation.
The harness evaluates and logs every revision, then selects the best incumbent among all versions
at the end; ties retain the existing incumbent. Replay improvement is development evidence,
not a measured guarantee of new live outcomes.
"""


def _messages(instructions, payload, max_input_allowance):
    # allow_nan=False makes malformed saved numerical records fail explicitly.
    content = instructions + "\nFULL_HISTORY_JSON\n" + json.dumps(payload, ensure_ascii=False, allow_nan=False)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    source_allowance(messages, max_input_allowance)
    return messages


def _context_record(record, pools):
    """Reference an identical already-included record instead of copying its source."""
    if record is None:
        return None
    def locate(value, path):
        if isinstance(value, dict):
            if value == record:
                return path
            for key, child in value.items():
                found = locate(child, path + [key])
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                found = locate(child, path + [index])
                if found is not None:
                    return found
        return None
    for name, pool in pools.items():
        found = locate(pool, [name])
        if found is not None:
            return {"full_record_path": found, "id": record.get("id") if isinstance(record, dict) else None}
    return record


def discovery_messages(parent, visible, suite_summary, prior_rollouts=(), scoring=TOTAL_EXTRA, *,
                       policy_features="block-v1", best=None, history=None, runtime_limits=None,
                       max_input_allowance=MAX_INPUT_ALLOWANCE):
    """Serialize all supplied prior histories and current observations, without omission.

    ``history`` replaces ``prior_rollouts`` when provided, avoiding accidental
    duplicate pools. ``best`` and ``parent`` are reference/context records supplied
    by the coordinator; the entire visible/current pool must also be supplied.
    Identical context records already in the pools use a full_record_path reference,
    so the baseline need only appear once without losing source or feedback.
    """
    if policy_features not in {"block-v1", "task-v1"}:
        raise ValueError("Unknown executable policy feature set")
    task = ""
    if policy_features == "task-v1":
        task = """Additional allowed attributes in task-v1: task_chat, task_qa, task_unknown, turn_index.
The task flags describe the declared serving category of the most recent request touching that
block; turn_index is its zero-based turn (QA is 0). Shared prefixes take the latest toucher.
There is no final-turn or continuation flag, and task category does not guarantee future reuse.
"""
    else:
        task = "block-v1 does not expose task flags or turn_index; do not access them.\n"
    pools = {"current_observations": list(visible),
             "completed_histories": history if history is not None else list(prior_rollouts)}
    return _messages(POLICY + task + objective_description(scoring) + "\n" + PUBLIC_PROGRAM,
                     {"program_mode": "executable", "policy_features": policy_features,
                      "selected_parent": _context_record(parent, pools),
                      "best_ever_training_policy": _context_record(best, pools),
                      **pools, "suite": suite_summary,
                      "runtime_limits": runtime_limits or {"wall_timeout_seconds": 2, "total_timeout_seconds": 120, "memory_mib": 256,
                                                           "max_output_bytes": 65536}},
                     max_input_allowance)


def controller_messages(current_version, feedback, histories, settings, revision_log=(), scoring=TOTAL_EXTRA, *,
                        online_rounds=None, objective=None, runtime_limits=None,
                        max_input_allowance=MAX_INPUT_ALLOWANCE):
    """Preserve the latest sequential base, every history and every revision verbatim."""
    for name in ("beta_calls", "beta_parallel"):
        if name not in settings and (not isinstance(objective, dict) or name not in objective):
            raise ValueError(f"Executable controller objective requires explicit {name}")
    if online_rounds is not None and (type(online_rounds) is not int or online_rounds < 1):
        raise ValueError("online_rounds must be a positive integer")
    return _messages(CONTROLLER + objective_description(scoring) + "\n" + PUBLIC_PROGRAM,
                     {"program_mode": "executable", "current_version": current_version,
                      "current_version_evaluation": feedback, "completed_histories": histories,
                      "settings": settings, "objective": objective,
                      "deployment": {"online_rounds_per_cycle": online_rounds,
                                     "workers": settings.get("workers"), "max_depth": settings.get("max_depth")},
                      "previous_revisions": list(revision_log),
                      "runtime_limits": runtime_limits or {"wall_timeout_seconds": 2, "total_timeout_seconds": 120, "memory_mib": 256,
                                                           "max_output_bytes": 65536}},
                     max_input_allowance)


def discovery_prompt(*args, **kwargs):
    """String convenience wrapper; transport adds the same system message."""
    return discovery_messages(*args, **kwargs)[1]["content"]


def controller_prompt(*args, **kwargs):
    """String convenience wrapper; transport adds the same system message."""
    return controller_messages(*args, **kwargs)[1]["content"]
