# Manual review of `answers.md`

The automatic verdicts in `answers.md` use a string match ("does the answer contain 0.95?"). That check cannot see wrong extra claims or wrong citations, so I read each answer next to the passages that were in its prompt. Run: `llama3.2:3b` (Q4_K_M), hybrid retrieval, top_k=5, temperature 0.

| id | automatic verdict | manual verdict | what happened | stage at fault |
|---|---|---|---|---|
| q01 | retrieval: page hit, evidence chunk missed | wrong (honest) | The chunk with the thresholds (`mmvp:p3:c17`, "cosine similarity exceeds 0.95 for CLIP ... less than 0.6 for DINOv2") ranked 9th in hybrid, outside top-5. The model correctly said the thresholds were not stated in its passages. It then echoed passage text into the answer, and copied the paper's own reference numbers `[8, 31, 71]` as citations. | retrieval (plus formatting issues in generation) |
| q02 | ok | correct, noisy | 9 patterns listed correctly. The list contains strings like `/sync` and `/sort-numeric-up`: icon glyphs from the PDF that pypdf extracted as text. | ingestion artefact |
| q03 | ok | partly wrong | The correct +10.7% (Interleaved MoF, Table 3) is there, but the answer also claims "6.5%". That number is from Table 2 (*Additive* MoF at ratio 0.5), which was in passage [1]. The model mixed up the two tables. | generation |
| q04 | ok | correct, imprecise wording | 346 images and 1129 questions are right; "1129 questions on average" is a wording error. | generation (minor) |
| q05 | ok | correct | 31.42%. | - |
| q06 | ok | correct | 591 / 538. | - |
| q07 | ok | correct | Pct. Diff and FP Ratio. | - |
| q08 | ok | correct | 256 bins, actions as tokens, 8 integers. | - |
| q09 | ok | correct facts, one wrong citation | 55B at 1-3 Hz on a multi-TPU cloud is right [1]. The extra claim "5B model at ~5 Hz" is also true, but it comes from passage [2] (`rt2:p6:c30`), not [1] as cited. | generation (citation) |
| q10 | ok | correct | No new motions; skills limited to the robot data distribution. | - |
| q11 | ok | correct | 46.9%, 12 tasks from 4 benchmarks. The "12 tasks" part is supported by [3]/[4] but not cited. | - (minor citation gap) |
| q12 | ok | correct | DDIM, 100 train / 10 inference iterations, 0.1 s on an RTX 3080. | - |
| q13 | ok | correct | Behaviour-cloning limits and higher latency/compute. | - |
| q14 | retrieval: page hit, evidence chunk missed | wrong (honest) | The Robomimic chunk (`diffusion_policy:p6:c44`) is mostly a flattened table and ranked 20th in hybrid. The model said the benchmarks were not stated. Again it emitted the paper's citation brackets `[12, 15, 29, 42]` as if they were passage numbers. | retrieval |
| q15 | ok | wrong | It says MMVP found GPT-4V "scoring below random guess level". The passage says the opposite: "Models **except GPT-4V and Gemini** scored below random guess level". The HallusionBench half (weak temporal reasoning) is supported. | generation (negation/scope error) |
| q16 | should have abstained | wrong | No passage gives GPT-4V's training settings; passage [1] gives RT-2's learning rate and batch size. The model abstained on the learning rate, then invented "batch size 8" and cited a Diffusion Policy hyperparameter table. | generation (hallucinated attribution) |
| q17 | correct abstention | abstained, with a false add-on | It abstained correctly, but then said the passages "contain information about ... the Open X-Embodiment benchmark". They do not. | generation (minor) |

## Summary

- 10 of 15 answerable questions are fully correct (q02, q05-q08, q10-q13 and, with a wording slip, q04). q09 is correct but has one mis-attributed citation.
- Retrieval failures: q01 and q14. In both cases the page was found but not the specific chunk with the answer, and the model correctly declined to make up the missing fact.
- Generation failures, where the evidence *was* in the prompt: q03 (mixed two tables), q15 (inverted "except"), q16 (invented a number for an unanswerable question). Two of the three are visible only on manual review; the string check passed them.
- Unanswerable questions: 1 of 2 abstained cleanly.
- Recurring formatting problem: when a passage contains the paper's own `[12, 15]` references, the 3B model sometimes copies them as citations. `parse_citations` flags these as `invalid_citations` instead of mapping them to wrong passages.
