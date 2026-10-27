# Model card: the weekly situation report

Sections follow Mitchell et al. (2019), *Model cards for model reporting*, shortened, with a section on replacing the
model.

**Code:** `twin/sitrep.py`; the prompt `prompts/sitrep_prompt.md`; the saved drafts `runs/sitreps/`. Notebook
sections 7.3 and 9.2.

## Model details
- An LLM, OpenAI's gpt-6-luna (reasoning effort low), drafts a brief for the operations officer, asked for at most
  150 words, from the week's facts as JSON: the work done, the teams, access, the reports held, the plan and the
  latest forecast's range. The restricted layers (the teams, and the access and security feed) are sent as counts only.
- The figure check reads every number in the draft and removes each sentence with a number that is not among the
  facts sent; if more than half the sentences go, the report is not shown, and the answer is "insufficient evidence".
  Without a key, a coded draft is written from the same facts.

## Intended use
A short weekly brief for the operations officer. Not for decisions; not for use outside the twin; it names no team or
closed zone, and gives no technical detail on munitions.

## Metrics
Sentences removed by the figure check; reports not shown; an expert's judgement of a sample.

## Evaluation data
The 52 weeks of seed 0's season under the main policy (section 9.2), 6 of them judged by Travis
(`runs/expert_review.csv`).

## Quantitative analyses (the full run)
- 52 drafts; no sentence removed by the figure check; none withheld.
- The 6 judged (weeks 5, 20, 28, 34, 47 and 49): every figure matches its week's facts (Claude's check). Travis
  judged weeks 34, 47 and 49 "Accurate and useful", and asked for more detail in weeks 5, 20 and 28 ("It's accurate,
  but it'll need much more technical detail" for week 5; "Great starting point" for weeks 20 and 28).
- Each of the 6 leaves out some of the week's facts: the teams active (in 4), the team-days worked (4), the areas still
  open (4), the mine areas found so far (3). The 6 run to 113 to 144 words; of all 52, 4 went over the 150-word
  limit (151 to 155 words), which nothing in the code enforces.

## Ethical considerations
The figure check tests the numbers, not what the words claim about them: week 47's draft ties the zones the access feed
closed to the plan, which the facts do not say. The restricted layers go to the model as counts only.

## Caveats and recommendations
A summary by design: the 150-word limit and the counts keep it short and inside the data rules, and leave out detail an
officer needs. Read it with the operations picture (section 7.1).

## Replacing this model
A new model or prompt changes the report's checksum, so the 52 weeks are drafted again; it is compared on the figure
check and on a fresh expert sample. More detail is a design change, not a model change: a longer limit and more of the
twin's facts in the prompt, with the restricted layers still sent as counts (section 11.5).

## Reference
Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T.
(2019). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and
Transparency* (pp. 220–229). Association for Computing Machinery. https://doi.org/10.1145/3287560.3287596
