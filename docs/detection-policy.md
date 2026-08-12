# Detection policy

This document is the specification for what the classifier blocks and why. It
is the authority: `utils/fast_detection.py` implements it, and
`tests/test_hybrid_voting.py` enforces the invariants it states.

## Design position

The system is an **input-side safeguard classifier**. It sits in front of a
language model and decides, per prompt, whether the request reaches the model
at all. It does not modify prompts, does not screen model outputs, and does not
attempt to make the underlying model safer. It is a filter, and it is the only
thing between the user and the assistant.

Two commitments follow from that position:

1. **The policy is explicit and versioned.** Thresholds live in one table in
   one file, not scattered across call sites. A change to what the system
   blocks is a reviewable diff.
2. **Recall is weighted above precision.** A missed jailbreak is a safety
   failure; a false positive is an annoyance. The voting rules are ordered so
   that several independent weak signals can block a prompt that no single
   model was confident about.

## Signals

Four independent signals score each prompt on [0, 1]. They are independent by
construction: a keyword list, two toxicity models trained on different corpora,
and a sentiment-derived safety score. Correlated signals would make the voting
rules below no better than a single model with a lower threshold.

| Stage | Model | Approx. latency | What it catches |
|-------|-------|-----------------|-----------------|
| Keyword | Pattern list plus regex | ~1ms | Direct instruction overrides, explicit harm requests |
| Toxicity | `unitary/toxic-bert` | ~100ms | Abusive and harmful phrasing |
| Hate speech | `martin-ha/toxic-comment-model` | ~150ms | Targeted hate content |
| Safety | `distilbert-base-uncased-finetuned-sst-2-english` | ~50ms | Adversarial framing and tone |

The transformer stages run concurrently on a thread pool, so ensemble latency
is bounded by the slowest stage rather than their sum.

### Early exit

A keyword score above `EARLY_EXIT_THRESHOLD` (0.45) blocks immediately and
skips the three transformer stages. The threshold sits deliberately below the
balanced high-confidence threshold of 0.50: a clear keyword hit is decisive on
its own, and spending ~300ms confirming it buys nothing. This is the main
latency and cost lever in the system.

### Dual-use dampening

The keyword stage is the easiest place to create false positives, because the
terms that appear in an attack also appear in legitimate questions about the
same subject. It therefore scores context, not just terms: a prompt matching a
harmful pattern is dampened when it also carries medical, educational,
gaming, professional-training, or software-engineering context.

"Instructions for making a gun" and "I am a certified instructor teaching a gun
safety course at a licensed range" contain the same trigger terms and must not
receive the same score. `tests/test_keyword_detection.py` asserts that ordering
directly.

## Voting rules

Rules are evaluated in order; the first to fire decides, and its name is
recorded in `voting_details` so any verdict can be attributed after the fact.
Scores are `[keyword, toxic, hate, safety]`.

**Rule 1 - High-confidence single signal.** Any one model above its confidence
threshold blocks. One model that is certain does not need a second opinion.

**Rule 2 - Multiple weak signals.** Two or more models above the weak-signal
threshold block, even though none is individually confident. This is the rule
that earns the ensemble its keep: sophisticated prompt injections tend to nudge
several detectors slightly rather than trip any one of them hard.

**Rule 3 - Majority consensus.** Two or more models above the standard 0.5
threshold block. Conventional ensemble voting.

**Rule 4 - Weighted ensemble.** The weighted sum of all four scores is compared
against a final threshold. Keyword carries the largest weight (0.40) because it
is the most precise signal for adversarial intent specifically, as opposed to
general unpleasantness. Weights sum to 1.0, which
`tests/test_hybrid_voting.py` asserts.

If no rule fires, the prompt is allowed.

## Sensitivity presets

| Threshold | high | balanced | conservative |
|-----------|------|----------|--------------|
| High-confidence (keyword, safety) | 0.40 | 0.50 | 0.80 |
| High-confidence (toxicity) | 0.50 | 0.60 | 0.80 |
| High-confidence (hate) | 0.40 | 0.50 | 0.70 |
| Weak signal | 0.15 | 0.20 | 0.40 |
| Weak signals required | 2 | 2 | 3 |
| Majority | 0.25 | 0.35 | 0.60 |
| Weighted ensemble | 0.20 | 0.30 | 0.55 |

Set with `FAST_DETECTION_SENSITIVITY`. `balanced` is the default; an
unrecognised value falls back to it with a warning rather than raising, so a
typo in an environment variable degrades to the default policy instead of
taking the service down.

### The monotonicity invariant

Every threshold is ordered `high <= balanced <= conservative`. This makes the
presets a genuine one-dimensional dial: raising sensitivity can only ever add
detections, never remove them.

This is enforced, not merely intended.
`test_high_sensitivity_never_allows_what_balanced_blocks` sweeps a grid of
score vectors and asserts that nothing blocked under `balanced` is allowed
under `high`.

The invariant is worth stating explicitly because an earlier revision violated
it: `balanced` had been tuned downward for subtle content without the other
presets being updated, leaving `high` uniformly *less* sensitive than
`balanced` on every threshold. The name promised more recall and the
configuration delivered less. The test exists so that cannot recur.

## Relationship to constitutional classifiers

Anthropic's constitutional classifiers work (Sharma et al., 2025) trains input
and output classifiers on synthetic data generated from a natural-language
constitution describing permitted and forbidden content. This system shares
part of that shape and is explicit about the part it does not.

Shared:

- An explicit written policy that decides what is blocked, kept separate from
  the models that produce the signals.
- Defence in depth: several independent detectors combined by stated rules,
  rather than one model's threshold.
- A deliberate, tunable position on the recall/precision trade-off, with the
  operating point exposed as configuration rather than buried in code.

Not shared, and not claimed:

- **No constitution-derived synthetic training data.** The policy here is a
  hand-authored threshold table, not a natural-language constitution compiled
  into training data. The transformer stages are off-the-shelf classifiers
  used as-is, not trained for this task.
- **No output-side classifier.** Screening is input-side only. A jailbreak that
  survives this filter is unconstrained thereafter; there is no streaming
  output check to catch a harmful completion in progress.
- **No adaptive adversarial evaluation.** The system has not been evaluated
  against an attacker who can query it and iterate. Published robustness
  results for constitutional classifiers come from red-teaming under exactly
  those conditions, and no comparable claim is made here.

The fine-tuning work in `src/models/` (DistilBERT, ELECTRA, and RoBERTa on
WildJailbreak, with deduplication and leakage verification in
`src/data/`) is the closest this repository comes to a task-trained classifier.
Those models are evaluated offline and are not currently wired into the serving
path, which uses the off-the-shelf ensemble described above.

## Known limitations

- **Keyword lists generalise poorly.** They are precise on known attack phrasing
  and blind to paraphrase. They are a fast first filter, not a defence.
- **Sentiment as a safety proxy is weak.** The safety stage uses an SST-2
  sentiment model; negative sentiment correlates with adversarial framing but
  is not the same thing, and the substitution is a latency compromise.
- **Thresholds are hand-tuned, not calibrated.** They were set by inspection on
  observed traffic. `src/evaluation/threshold_calibration.py` is the starting
  point for deriving them from a labelled set instead.
- **English only.** Every stage is English-trained; non-English attacks are
  effectively unscreened.
- **No per-user or per-session state.** Each prompt is judged in isolation, so
  an attack split across several benign-looking turns is not detected.
