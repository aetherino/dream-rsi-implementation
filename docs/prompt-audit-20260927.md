# Prompt audit after the 240-call experiment

Source: `runs/comparison-long-realtime-20260927`. This audit uses saved public
programs, rationales, prompts, and simulator feedback. It does not inspect hidden
reasoning and does not claim that a revised prompt has already improved outcomes.

## Evidence

There were 221 policy proposals and 19 controller proposals. Of the policy
proposals, 220 contained a retention-expression string. **55 expressions repeated
an earlier exact expression within the same run** (25% of these strings). This
counts repeated submissions beyond the first, not pairs of duplicates, and does
not attempt semantic equivalence. Some repeats are predictable from the context
design: discovery sees only the recent 12 nodes and the best node from the last
four rollouts, not a complete list of attempted expressions. The global best can
also disappear from the prompt after enough rollouts, even though the harness
still retains it for final selection.

**Nine of 221 proposals failed**:

| Failure | Count |
| --- | ---: |
| Numeric constants exceed the stated bound | 3 |
| Extra JSON fields outside the schema | 4 |
| Invalid JSON | 1 |
| Expression exceeds the AST size limit | 1 |

One rejected program explicitly said its unnecessary nested conditionals were
added to satisfy the depth-12 requirement. The actual rule is an upper bound, not
a required depth. The current prompt says "at most", so this is an observed
instruction-following error; clearer wording may help but is not a proven fix.

Other public rationales reveal misconceptions:

- A proposal justified a huge positive depth coefficient by claiming eviction of
  a deep block requires recomputing its descendants. Eligible victims are leaves,
  which have **no resident children**. Removing a leaf keeps its ancestors cached;
  depth alone is not a direct per-eviction recomputation-cost multiplier.
- A proposal added `now / 100000000` to improve tie handling. At a given eviction,
  `now` is the same for every eligible entry, so adding that common term does not
  change ranking (and its constant violates the DSL bound).
- Some controller rationales describe priorities opposite to their formulas, or
  describe multiple simultaneous refinements of the same leaf. Selection is by
  descending numeric priority over distinct nodes; a leaf can be selected once.
- Some rationales invent incorrect suite totals. For example, one claimed an
  LRU denominator of 83,808; the frozen training denominator is **83,312**. The
  evaluator computes the actual score correctly. Model prose is not evidence.

Controller prompting also has a concrete context omission: the supplied settings
describe **six offline replay rounds**, but do not explicitly state the actual
**three online discovery rounds per cycle**. Those are different horizons. The
historical trees contain three-round observations, but the model should not have
to infer the deployment limit from them.

## Proposed prompt/context revision for a later controlled test

1. State that depth and AST limits are maximums, not targets. Require exactly the
   allowed JSON keys, small expressions, and bounded constants. Keep rationales
   to one or two sentences rather than long unsupported explanations.
2. Include a worked leaf-eviction example: for a resident chain A → B → C, only C
   is eligible; evicting C leaves A and B reusable. Distinguish immediate removal
   from later evictions and future reuse. Avoid claiming all shallow or deep
   entries are inherently preferable.
3. Supply authoritative aggregate saved tokens, denominator, and score directly
   in candidate feedback. Label expected improvements as hypotheses.
4. Include a compact run-wide list of attempted expressions and their scores,
   plus the best-ever training policy. Ask for a new expression or a clearly
   justified retest. Avoid silently charging repeated formulas as fresh ideas.
5. Give controller proposals a small explicit ranking example, the distinct-node
   constraint, and separate online/replay round limits. Ask the model to check
   whether its numeric priorities actually implement its stated behavior.

Any prompt revision should be versioned and compared against the existing prompt
with the same controller, model settings, call budget, and fresh evaluation data.
The current follow-up deliberately holds prompts constant to isolate longer
search and controller transfer. The MiMo thinking setting is also held disabled.

Prompt improvements will not by themselves resolve replay overfitting or a
miscalibrated search-cost objective. At `beta_calls=0.02`, one extra revealed call
costs two percentage points of task score before the small parallelism bonus.
This can favor pruning branches that might produce useful future improvements.
The prospective controller comparisons test that concern directly.
