# Evaluation Spec — Call of Duty Review Classifier

## Purpose

This spec defines how model outputs are evaluated for the three-class CoD review classifier. It exists to make grading decisions reproducible and to document edge cases encountered during annotation.

---

## Label Definitions (Grading Version)

These are the operative definitions used when scoring model predictions against ground truth.

### `analysis`
A review that evaluates the game across at least two distinct dimensions (e.g., campaign + multiplayer, graphics + sound, mechanics + pacing) and provides reasoning or comparison — not just co-occurrence of topic words in a single sentence. Length alone does not determine this label; a 200-word review that just repeats "I love this game" in different ways is still `reaction`.

### `reaction`
A review that expresses a personal emotional response — positive or negative — without structured multi-dimensional reasoning. Includes: single-sentence reviews, truncated "… Expand" entries, reviews that mention multiple topics but make no comparative or evaluative claim across them.

### `hot_take`
A review that makes a generalizing, provocative, or contrarian claim directed at the game's or franchise's reputation — one that asserts something beyond the reviewer's personal experience. The distinguishing move: the reviewer is making a claim *about the franchise or the community*, not just describing their own reaction. Common markers include franchise-level indictments ("this is the same game every year"), appeals to other community members ("you people are paying $60 for DLC"), blame attributed to developer/publisher decisions (SBMM, microtransactions, Activision greed), or rhetorical devices like anaphora used to persuade rather than describe.

---

## Boundary Cases

| Case | Decision | Reasoning |
|------|----------|-----------|
| Review mentions campaign and multiplayer in one sentence with no elaboration | `reaction` | Co-occurrence is not evaluation. |
| Review criticizes price but doesn't generalize to franchise | `reaction` | "This isn't worth $60" is personal; "You're paying $60 for DLC every year" is `hot_take`. |
| Review uses structured numbered list (1. Graphics 2. Multiplayer) | `analysis` | Explicit framework = analysis even if each item is brief. |
| Review says "best CoD ever" with no elaboration | `reaction` | No evaluation dimension present. |
| Review says "best CoD ever" and contrasts with current franchise direction | `hot_take` | The contrast is a franchise-level claim. |
| Non-English review (Russian, etc.) | Apply same criteria to available content | Do not default to `reaction` for language alone. |
| Truncated "… Expand" entry | `reaction` | No accessible content to evaluate. |

---

## Confidence Threshold

For the sample classifications table in the README, "high confidence" is defined as softmax probability ≥ 0.80 for the predicted class. "Medium" is 0.60–0.79. "Low" is below 0.60. Predictions below 0.60 on the majority class are flagged for manual review.

---

## Metrics Priority

Evaluation prioritizes in this order:

1. **Macro F1** — because class imbalance is severe (`hot_take` at 3.5%), accuracy alone is misleading.
2. **Per-class F1** — especially for `hot_take`, which is the most theoretically interesting class and the hardest to learn.
3. **Overall accuracy** — reported but not the primary signal.

A model that achieves 66% accuracy by predicting `reaction` for everything is considered a failure by this spec. The baseline bar is: per-class F1 for `analysis` > 0.50, with any non-zero `hot_take` recall considered a meaningful result given the 7-example support.

---

## Known Data Quality Issues

- Seed labels at indices 4 and 6 were marked `reaction` by the notebook but could reasonably be `analysis`. These are preserved as-is and treated as ground truth for training; they may introduce noise near the analysis/reaction boundary.
- Index 7 was marked `analysis` by the notebook seed but is arguably `reaction` (single-sentence, no comparative reasoning). Preserved as-is.
- `hot_take` has only 7 training examples after the 70/15/15 split, leaving ~4-5 training examples for that class. F1 scores for `hot_take` should be interpreted with extreme caution.

---

## AI Assistance Disclosure (Eval Spec)

This spec was drafted with AI assistance (Claude, claude-sonnet-4-6). The boundary case table and confidence threshold definition were generated from the label definitions I supplied and then reviewed and edited for accuracy against the actual dataset content.
