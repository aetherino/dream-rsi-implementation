"""Small numeric language. Generated source is interpreted, never eval/exec'd."""
import ast
import math
import operator

from cache_sim.policies import Policy

POLICY_NAMES = frozenset("now depth inserted_at last_access frequency insertion_order access_order".split())
GLOBAL_NAMES = frozenset("round_index attempts_seen branches_opened best_score".split())
LEAF_NAMES = GLOBAL_NAMES | frozenset("score gain depth stagnation failure_streak valid age_rounds".split())
FUNCTIONS = {"min": (min, 2), "max": (max, 2), "abs": (abs, 1),
             "log1p": (math.log1p, 1), "sqrt": (math.sqrt, 1)}
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
           ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}


class Expression:
    def __init__(self, source, names):
        if not isinstance(source, str) or not 0 < len(source) <= 1000:
            raise ValueError("Expression must contain 1–1000 characters")
        try:
            self.tree = ast.parse(source, mode="eval").body
        except (SyntaxError, RecursionError) as exc:
            raise ValueError("Invalid expression syntax") from exc
        if sum(1 for _ in ast.walk(self.tree)) > 96:
            raise ValueError("Expression exceeds 96 AST nodes")
        self._validate(self.tree, names, 0)
        self.source = source

    def _validate(self, node, names, depth):
        if depth > 12:
            raise ValueError("Expression nesting exceeds 12")
        children = []
        if isinstance(node, ast.Constant):
            if type(node.value) not in (bool, int, float) or not math.isfinite(node.value) or abs(node.value) > 1e6:
                raise ValueError("Only finite numeric constants of magnitude <= 1e6 are allowed")
        elif isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            children = [node.left, node.right]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
            children = [node.operand]
        elif isinstance(node, ast.Compare) and all(type(op) in COMPARE for op in node.ops):
            children = [node.left, *node.comparators]
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            children = node.values
        elif isinstance(node, ast.IfExp):
            children = [node.test, node.body, node.orelse]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
            if node.keywords or len(node.args) != FUNCTIONS[node.func.id][1]:
                raise ValueError("Wrong function arity")
            children = node.args
        else:
            raise ValueError(f"Unsupported expression construct: {type(node).__name__}")
        for child in children:
            self._validate(child, names, depth + 1)

    def __call__(self, values):
        def visit(node):
            if isinstance(node, ast.Constant):
                result = node.value
            elif isinstance(node, ast.Name):
                result = values[node.id]
            elif isinstance(node, ast.BinOp):
                result = BINARY[type(node.op)](visit(node.left), visit(node.right))
            elif isinstance(node, ast.UnaryOp):
                result = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}[type(node.op)](visit(node.operand))
            elif isinstance(node, ast.Compare):
                left = visit(node.left)
                result = True
                for op, right_node in zip(node.ops, node.comparators):
                    right = visit(right_node)
                    if not COMPARE[type(op)](left, right):
                        result = False
                        break
                    left = right
            elif isinstance(node, ast.BoolOp):
                result = (all(bool(visit(n)) for n in node.values) if isinstance(node.op, ast.And)
                          else any(bool(visit(n)) for n in node.values))
            elif isinstance(node, ast.IfExp):
                result = visit(node.body if visit(node.test) else node.orelse)
            else:
                result = FUNCTIONS[node.func.id][0](*(visit(n) for n in node.args))
            if not math.isfinite(result) or abs(result) > 1e12:
                raise ValueError("Expression produced an out-of-range number")
            return result
        try:
            return visit(self.tree)
        except (ArithmeticError, KeyError, TypeError) as exc:
            raise ValueError("Expression arithmetic failed") from exc


def metadata(spec):
    if not isinstance(spec, dict) or not isinstance(spec.get("name"), str) or not 0 < len(spec["name"]) <= 80:
        raise ValueError("Program requires a name of 1–80 characters")
    if not isinstance(spec.get("rationale", ""), str) or len(spec.get("rationale", "")) > 4000:
        raise ValueError("Rationale must be text of at most 4000 characters")


class RetentionPolicy(Policy):
    def __init__(self, spec):
        metadata(spec)
        if set(spec) - {"name", "rationale", "retention_score"}:
            raise ValueError("Unexpected policy fields")
        self.name = spec["name"]
        self.expression = Expression(spec.get("retention_score"), POLICY_NAMES)

    def choose(self, eligible, now):
        def key(entry):
            values = {name: getattr(entry, name) for name in POLICY_NAMES if name != "now"}
            values["now"] = now
            return self.expression(values), entry.access_order
        return min(eligible, key=key).block_id


LRU_SPEC = {"name": "initial-lru", "retention_score": "last_access", "rationale": "LRU baseline"}
INITIAL_CONTROLLER = {"name": "parallel-refinement", "root_if": "True", "root_priority": "1",
                      "leaf_if": "failure_streak < 2 and stagnation < 3",
                      "leaf_priority": "2 + score - 0.1 * depth"}


class Controller:
    def __init__(self, spec):
        metadata(spec)
        fields = {"root_if", "root_priority", "leaf_if", "leaf_priority"}
        if set(spec) - fields - {"name", "rationale"}:
            raise ValueError("Unexpected controller fields")
        self.expr = {key: Expression(spec.get(key), GLOBAL_NAMES if key.startswith("root") else LEAF_NAMES)
                     for key in fields}

    def select(self, nodes, round_index, workers, max_depth, exhausted=()):
        """Only revealed observations enter this function; IDs break ties only."""
        if not nodes or nodes[0]["id"] != 0:
            raise ValueError("Missing initial root")
        by_id = {n["id"]: n for n in nodes}
        parents = {n["parent"] for n in nodes[1:]}
        globals_ = {"round_index": round_index, "attempts_seen": len(nodes) - 1,
                    "branches_opened": sum(n["parent"] == 0 for n in nodes[1:]),
                    "best_score": max(n["score"] for n in nodes if n["valid"])}
        ranked = []
        if 0 not in exhausted and self.expr["root_if"](globals_):
            ranked.append((self.expr["root_priority"](globals_), 0))
        for node in nodes[1:]:
            if node["id"] in parents or node["id"] in exhausted or node["depth"] >= max_depth:
                continue
            parent = by_id[node["parent"]]
            values = globals_ | {k: node[k] for k in ("score", "depth", "stagnation", "failure_streak", "valid")}
            values.update(gain=node["score"] - parent["score"], age_rounds=round_index - node["round"])
            if self.expr["leaf_if"](values):
                ranked.append((self.expr["leaf_priority"](values), node["id"]))
        return [node_id for _, node_id in sorted(ranked, key=lambda x: (-x[0], x[1]))[:workers]]
