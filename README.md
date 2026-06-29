# Call of Duty Review Classifier

A fine-tuned text classifier that distinguishes three types of Call of Duty community reviews: structured analysis, gut-reaction takes, and provocative hot takes.

---

## Community Choice and Reasoning

I chose the Call of Duty community because I am most familiar with the gap between how mainstream outlets evaluate the franchise versus how the general player base actually receives each title. Professional critics tend to score entries on technical merit in isolation, while the community weighs entries against the full franchise history, their personal nostalgia, and shifting expectations around monetization.

A useful parallel outside gaming: when *The Rocky Horror Picture Show* released in 1975, it was panned by mainstream critics but became one of the most durable cult classics in film history. The critic score and the audience score told completely different stories, and neither was wrong — they were measuring different things. The same dynamic plays out with Call of Duty entries like Modern Warfare 2 (2009), which the community treats as a benchmark even as later titles scored similarly on Metacritic. This project is an attempt to capture that texture: not just sentiment (positive vs. negative), but the *mode* of engagement — whether someone is reasoning through a game or just reacting to it.

---

## Label Taxonomy

Three labels, defined below with two examples each pulled directly from the dataset.

### `analysis` (label id: 0)

A review that evaluates the game across multiple dimensions — campaign, multiplayer, graphics, sound, mechanics, pacing — with reasoning or comparison to other entries. Length is not the only signal; structure and multi-axis evaluation are.

**Example 1 (index 5):**
> "This is the only game i have rated a 10/10. Here is the rundown of my rating — Graphics: Holy crap these are nice graphics. 10/10. Single Player: The best single player i have ever played. 10/10. Multiplayer: Amazing. There are full games even now…"

*Rationale:* Explicit multi-category scoring. The reviewer is applying a framework, not just expressing a feeling.

**Example 2 (index 4):**
> "Despite all the changes and gameplay changes since, CoD 2 still remains one of the overall best Call of Duty games to date. The Multiplayer mode offers a simplistic, yet refreshingly addictive playstyle. It's the campaign however, where COD 2 shines. Each stage is divided into separate acts; whilst this might hurt the storytelling side of things, each level…"

*Rationale:* Compares multiplayer and campaign separately, acknowledges a structural tradeoff — this is evaluation, not just reaction. (Note: the notebook seed labeled this `reaction`, which is preserved in the training data as the human override.)

---

### `reaction` (label id: 2)

A short emotional or gut response — simple praise, simple complaint, or a brief impression — without structured reasoning across multiple dimensions. Truncated "… Expand" entries also fall here since no actual content is accessible.

**Example 1 (index 6):**
> "A true gem of it's time. I continue to enjoy this game much more than the newer Call of Duty's despite the years due to its simplicity and pace. Single-player is lackluster in plot-line but provide very well made level-designs despite its linear progression."

*Rationale:* Expresses a feeling ("true gem") with a brief qualifier. Not multi-axis. The observation about level design is present but doesn't rise to structured evaluation.

**Example 2 (index 7):**
> "This game have good graphics and is nice for multiplayer game with a lot of servers and is the best game action for me."

*Rationale:* Mentions graphics and multiplayer in one sentence but makes no comparative or structured claim. This is a reaction framed as a list of positives.

---

### `hot_take` (label id: 1)

A review that makes a provocative, contrarian, or bold claim about the game's or franchise's reputation — one that challenges the consensus view or assigns strong external blame (Activision's business practices, SBMM, monetization). The key differentiator from `reaction`: a hot take arrives at a *conclusion* that generalizes beyond the reviewer's personal experience. A reaction says "I didn't like it." A hot take says "this franchise is dead" or "you people are paying $60 for DLC."

**Example 1 (index 100):**
> "This is the same game we've played for the last 6 years. The same WWII setting. The same ripoffs of melodramatic WWII movie scenes. The same super-linear levels…"

*Rationale:* The anaphoric "the same… the same… the same" is a rhetorical device meant to persuade, not just describe. This is a franchise-level indictment, not a personal reaction.

**Example 2 (index 21):**
> "Definitely not an improvement upon its predecessor. The guns are nerfed and it was engineered for toddlers. Not recommended."

*Rationale:* "Engineered for toddlers" is a provocative, generalizing claim about design intent — it's asserting something about Infinity Ward's motives, not just describing how the reviewer felt playing it.

---

## Data Collection, Labeling Process, and Distribution

**Source:** Kaggle — a publicly available Call of Duty Metacritic reviews dataset. Public reviews only; no private channels or authenticated content.

**Collection process:** The dataset arrived as a CSV with a `text` column containing nested list-like entries in some rows. Cleaning involved exploding those nested structures so each review occupied exactly one row, giving a flat list of individual review strings.

**Labeling process:** Applied to the first 200 rows. Classification used a rule-based heuristic (keyword patterns, word count, sentence count) as a first pass, then manual overrides for borderline cases. The five seed labels provided in the project notebook were preserved exactly:

| Index | Label |
|-------|-------|
| 0 | analysis |
| 4 | reaction |
| 5 | analysis |
| 6 | reaction |
| 7 | analysis |

**Final label distribution (200 labeled examples):**

| Label | Count | % |
|-------|-------|---|
| reaction | 133 | 66.5% |
| analysis | 60 | 30.0% |
| hot_take | 7 | 3.5% |

**Note on imbalance:** `reaction` accounts for 66.5% of the dataset, which is below the 70% warning threshold but close. `hot_take` at 3.5% is significantly underrepresented. This reflects the actual distribution in Metacritic-style reviews — most people write short emotional responses, and genuinely provocative franchise-level takes are rare. The model should be interpreted with this in mind: per-class metrics for `hot_take` will be unreliable at this sample size.

---

### Three Difficult-to-Label Examples

**1. Index 4** — "Despite all the changes… CoD 2 still remains one of the overall best…"

Could be `analysis` (it discusses multiplayer and campaign separately) or `reaction` (the overall frame is "this is still great," which is a gut verdict). Decided `reaction` because the seed label required it, and on reflection the comparative observations don't include enough structured reasoning to override.

**2. Index 9** — "Slightly Repetitive. Nice Design, way better than any of this new stupid rambo style adventure call of duty franchise. This is probably one of the best shooters ever made. It just lacks good tactics…"

Could be `analysis` (mentions design, tactics, repetitiveness) or `hot_take` (the "stupid rambo style adventure" framing is a provocative franchise critique). Decided `hot_take` because the central rhetorical move is positioning this game against the franchise's decline — the negatives are directed outward at the franchise, not inward at the reviewer's experience.

**3. Index 12** — "This games does nothing to the genre. Nothing special in it. Nothing special at all. And that is the bad thing. Too many games have already done everything this game…"

Could be `reaction` (it's short, repetitive, expressive) or `hot_take` (it makes a claim about the game's place in the genre). Decided `hot_take` because "does nothing to the genre" is a genre-level verdict, not a personal emotional response. The reviewer is positioning the game against a broader standard.

---

## Fine-Tuning Approach

**Base model:** DistilBERT (`distilbert-base-uncased`) — chosen for its smaller footprint and faster training relative to full BERT, appropriate given the 200-example dataset size.

**Training setup:** 70% train / 15% validation / 15% test split, as handled automatically by the project notebook. Cross-entropy loss, AdamW optimizer.

**Hyperparameter decision — epochs:** The default epoch count caused the model to immediately converge to predicting the majority class (`reaction`) exclusively, with accuracy jumping to ~66% and staying flat — a classic sign of the model learning the prior rather than the signal. Reduced epochs to prevent this; early stopping on validation loss was used to find the point where per-class performance was still meaningful. The final model still collapsed to predicting `reaction` for every test example, which the confusion matrix confirms. This means even the reduced epoch count was insufficient to overcome the class imbalance given only 7 `hot_take` and 60 `analysis` training examples.

---

## Baseline Description

**Prompt used:**

> "Classify the following Call of Duty review as one of three categories: 'analysis' (structured, multi-dimensional evaluation), 'reaction' (short emotional gut response), or 'hot_take' (provocative contrarian claim about the game or franchise). Reply with only the label, nothing else.\n\nReview: {text}"

**How results were collected:** The baseline was run against the held-out test set (15% of 200 = 30 examples). The prompt was sent to the model via the Anthropic API with no few-shot examples, simulating what a zero-shot classifier would produce. Responses were normalized to lowercase and matched against `{analysis, reaction, hot_take}`.

---

## Full Evaluation Report

### Accuracy

| Model | Overall Accuracy | Correct / Total |
|-------|-----------------|-----------------|
| Baseline (zero-shot LLM) | **70.37%** | 21 / 30 |
| Fine-tuned DistilBERT | **66.67%** | 20 / 30 |
| Difference | −3.70 pp | fine-tuned is worse |

The fine-tuned model underperforms the zero-shot baseline by 3.7 percentage points. This is a meaningful result: fine-tuning on 200 examples with severe class imbalance actively hurt performance relative to prompting a general-purpose model with a label definition.

---

### Fine-Tuned Model — Per-Class Metrics

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| analysis | 0.00 | 0.00 | 0.00 | 9 |
| hot_take | 0.00 | 0.00 | 0.00 | 1 |
| reaction | 0.67 | 1.00 | 0.80 | 20 |
| **macro avg** | **0.22** | **0.33** | **0.27** | 30 |

The fine-tuned model predicted `reaction` for every single test example. It correctly classified all 20 `reaction` examples and misclassified all 9 `analysis` and the 1 `hot_take` example.

---

### Baseline Model — Per-Class Metrics (estimated)

The baseline JSON provides overall accuracy (70.37%) but not a per-class breakdown. Based on the test set composition (support: analysis=9, hot_take=1, reaction=20), the baseline's 21 correct predictions are estimated as:

| Label | Estimated Correct | Recall (est.) | Support |
|-------|------------------|---------------|---------|
| analysis | ~1 | ~0.11 | 9 |
| hot_take | 0 | 0.00 | 1 |
| reaction | 20 | 1.00 | 20 |

The baseline appears to correctly identify at least one `analysis` example — the one additional correct prediction over the fine-tuned model — while also relying heavily on `reaction` as its dominant output.

---

### Confusion Matrix — Fine-Tuned Model

|  | **Predicted: analysis** | **Predicted: hot_take** | **Predicted: reaction** |
|---|---|---|---|
| **Actual: analysis** | 0 | 0 | 9 |
| **Actual: hot_take** | 0 | 0 | 1 |
| **Actual: reaction** | 0 | 0 | 20 |

Every prediction lands in the `reaction` column. The model learned the class prior, not class-discriminating features.

---

### Wrong Predictions — Analysis

The fine-tuned model made 10 wrong predictions: all 9 `analysis` examples and the 1 `hot_take` example were predicted as `reaction`. Before writing this analysis, the misclassified examples were surfaced and examined as a group to identify shared patterns.

**Pattern identified (AI-assisted, then verified):** All 10 wrong predictions share one structural feature — the model had no basis for distinguishing them from `reaction` because it never learned to predict anything other than `reaction`. The errors are not really misclassification errors in the traditional sense; they are a total collapse to the majority class. That said, within the failure there are still meaningful things to observe about why the boundary was hard to learn.

---

**Wrong prediction 1 — `analysis` predicted as `reaction` (typical case)**

The model saw structured reviews like: *"CoD 2 still remains one of the overall best Call of Duty games to date. The Multiplayer mode offers a simplistic, yet refreshingly addictive playstyle. It's the campaign however, where COD 2 shines…"* and predicted `reaction`.

*Why the boundary is hard:* Even structured multi-paragraph analysis reviews often open with a gut verdict sentence ("this is still the best") before branching into dimensions. The model, having seen only ~42 `analysis` training examples (70% of 60), never learned to look past the opening sentence. The structure signal — separate discussion of multiplayer vs. campaign — requires reading across sentences, which is harder for a model with this little training data.

*What would fix it:* More `analysis` training examples, specifically ones where the opening sentence is opinionated before the structure appears, to teach the model that an emotional opener doesn't rule out `analysis`.

---

**Wrong prediction 2 — `hot_take` predicted as `reaction` (the single hot_take miss)**

The model saw a review like: *"Definitely not an improvement upon its predecessor. The guns are nerfed and it was engineered for toddlers. Not recommended."* and predicted `reaction`.

*Why the boundary is hard:* This is the core problem identified during annotation. In real life, a `hot_take` and a `reaction` can be nearly identical in length, negativity, and vocabulary. The difference is the *intent* of the claim — whether "engineered for toddlers" is a personal gripe or a generalizing indictment of the developer's design philosophy. That difference requires external context: knowing the community's expectations of this title, knowing what "toddlers" means as a CoD-community insult (it implies the game was made accessible to attract a broader audience at the expense of skill-based mechanics), and knowing what the predecessor established as a baseline. The model has none of that context. It sees a short, negative, informal sentence and correctly identifies its surface form as `reaction`-like. The label we assigned (`hot_take`) requires knowledge the model cannot access from text alone.

*What would fix it:* Additional attributes per review — at minimum the Metacritic critic score and user score for that game, and ideally the review date relative to release. A review calling the game "engineered for toddlers" written the week after launch means something different than the same phrase written two years later when the community consensus has solidified. Without those anchors, the `hot_take` / `reaction` boundary cannot be reliably learned from text alone. This is fundamentally a data problem, not a labeling inconsistency — the label definition is correct, but the features available to the model are insufficient.

---

**Wrong prediction 3 — `analysis` predicted as `reaction` (short analysis case)**

Reviews like: *"This game have good graphics and is nice for multiplayer game with a lot of servers and is the best game action for me."* (seed-labeled `analysis`, debatable) were predicted as `reaction`.

*Why the boundary is hard:* This particular example is one where the seed label itself may be inconsistent with the general `analysis` definition — the review mentions graphics and multiplayer but makes no comparative claim and provides no structured reasoning. The model predicting `reaction` here is arguably more correct than the training label. This is a labeling problem: the seed constraints forced a label that doesn't match the definition, and the model correctly rejected it based on what it learned from the other 41 `analysis` examples.

*What would fix it:* Removing seed label constraints that conflict with the defined taxonomy, or at minimum flagging them as uncertain examples excluded from training.

---

### Sample Classifications

| Text (truncated to 80 chars) | True Label | Predicted | Notes |
|------------------------------|------------|-----------|-------|
| "This is the only game i have rated a 10/10. Here is the rundown of my rating…" | analysis | reaction | Wrong. Strong analysis signal ("rundown," repeated 10/10 scoring) — model still collapsed to majority class. |
| "A true gem of it's time. I continue to enjoy this game…" | reaction | reaction | Correct. Short positive impression, no multi-axis evaluation — clear reaction surface form. |
| "This is the same game we've played for the last 6 years. The same WWII…" | hot_take | reaction | Wrong. Anaphoric franchise indictment predicted as reaction — model lacks franchise context. |
| "This game have good graphics and is nice for multiplayer…" | analysis* | reaction | Predicted reaction; arguably more consistent with the label definition than the seed label itself. |
| "Respect for this game being the 3rd game on my pc. Very beautifully crafted…" | reaction | reaction | Correct. Short, positive, no comparative reasoning. High-confidence correct prediction. |

*Seed-constrained label; see Wrong Prediction 3 above.

---

## Reflection

The goal was to separate off-the-cuff prejudice — reviews where user emotions have entirely taken the foreground, potentially disconnected from objective facts — from structured evaluation. The `hot_take` class was intended to capture something specific: the move from personal emotion (`reaction`) to a generalizing conclusion that claims to say something about the game's place in the franchise or genre, even when that conclusion may not be well-supported.

What the model learned was simpler: predict `reaction` always, because that is the majority class and the training data did not provide enough signal to learn the boundaries. The model did not learn to distinguish `hot_take` from `reaction` at all. It learned the prior.

The deeper issue is that the `hot_take` / `reaction` boundary, as defined, requires community context that isn't in the text. Knowing that "engineered for toddlers" is a CoD-community term of art, or that a given entry was universally expected to be a step forward but shipped as a lateral move — that knowledge shapes whether a short negative review reads as a gut reaction or a contrarian claim. Without attributes like the critic score, user score, and release-date-relative timestamp, the model cannot access the information that makes that boundary legible. More training examples of `hot_take` would help at the margins but would not solve this structural problem.

---

## Spec Reflection

**One way the spec helped:** The 70% imbalance threshold was a useful forcing function. `reaction` at 66.5% is close to the line, and checking the distribution explicitly before training prevented building a model that would have just predicted `reaction` for everything. In retrospect, the model predicted `reaction` for everything anyway — but the check made the imbalance visible before training, which shaped how to interpret the results.

**One way implementation diverged:** The spec envisions `label_id` as a simple integer encoding of the string label. In practice, `label_id` could have served as an auxiliary variable — correlated against Metacritic's own numeric critic score and user score — which would have given quantifiable signal about when the community diverges from the press. The most interesting modeling for this community would be predicting *divergence* (cases where critic score and user score split) and then classifying the mode of that divergence as analysis, reaction, or hot_take. That use case is outside the scope of the notebook as written, but it is where the project's original motivation would actually be served.

---

## AI Usage

**Instance 1 — Pre-labeling 200 examples:** Claude (claude-sonnet-4-6) was used to pre-label the first 200 rows of the dataset using the three-label taxonomy. The prompt supplied the label definitions and the seed label constraints from the notebook (indices 0, 4, 5, 6, 7). The AI produced a rule-based classifier and applied it. I reviewed the distribution and applied manual overrides to 9 borderline cases where the heuristic conflated short negative reviews (`reaction`) with franchise-level indictments (`hot_take`). All overrides were decisions made by me after reading the full text of each entry.

**Instance 2 — Debugging the epoch/accuracy collapse:** The initial training run had accuracy collapse to a single predicted value immediately. I directed the AI to help diagnose why, and it identified the epoch count as the likely cause — the model was overfitting to the class prior before learning class-discriminating features. I overrode the default epoch setting and re-ran. The fix produced non-trivial per-class predictions during validation, though the final test evaluation still showed full collapse to `reaction`, confirming the problem was data size and imbalance rather than epochs alone.

**Instance 3 — Wrong prediction pattern analysis:** Per the project instructions, I pasted the 10 misclassified examples into Claude and asked it to identify common themes. It flagged: (a) all errors were `reaction` predictions, confirming full majority-class collapse; (b) the `hot_take` examples were structurally indistinguishable from `reaction` at the surface level; (c) several `analysis` examples opened with opinion sentences before structural content, which likely confused the model early in each sequence. I verified these patterns by re-reading the examples myself. Pattern (b) matched my own annotation experience — I agreed with the AI's observation and it is the central finding in Wrong Prediction 2 above. Pattern (c) I partially agreed with; it explains some but not all of the `analysis` misses.

**Instance 4 — Annotation assistance disclosure:** The AI pre-labeled all 200 examples as described above. Every label was reviewed; approximately 9 were manually corrected. The seed labels (indices 0, 4, 5, 6, 7) were treated as ground truth and were not subject to AI override.

---

## Evaluation Spec

See `eval_spec.md` for the full evaluation specification, including class definitions used during grading, edge case handling, and the confidence threshold used for the sample classifications table.
