# Ask VEPai golden dataset

40 real-world VEP questions modelled on patterns from `Ensembl/ensembl-vep` GitHub issues and public Ensembl helpdesk threads. Used for retrieval evaluation (MRR, top-1, top-5) and answer faithfulness scoring (RAGAS) of the Ask VEPai RAG prototype.

## Coverage

| category | count |
|---|---|
| syntax_cli | 1 |
| input_output_formats | 4 |
| cache_management | 4 |
| regulatory_variants | 1 |
| clinvar_annotation | 1 |
| conservation_scores | 1 |
| plugins | 5 |
| filtering | 2 |
| performance_tuning | 1 |
| rest_api | 3 |
| consequence_types | 4 |
| hgvs_notation | 2 |
| annotations | 8 |
| species_assembly | 2 |
| installation | 2 |

## Difficulty distribution

- easy: 16 (40%)
- medium: 16 (40%)
- hard: 8 (20%)

## Schema

Each question object:

| field | required | type | notes |
|---|---|---|---|
| `id` | yes | string | stable identifier `vep-q-NNN` |
| `question` | yes | string | natural-language user question |
| `category` | yes | string | topic bucket (14 possible values) |
| `subcategory` | no | string | finer-grained tag |
| `difficulty` | yes | enum | `easy` \| `medium` \| `hard` |
| `interface` | yes | list | one or more of `cli`, `web`, `rest`, `docker` |
| `vep_version_min` | yes | string | Ensembl release this question applies to |
| `expected_answer_keywords` | yes | list | strings that should appear in a correct answer (used for lexical scoring) |
| `expected_answer_summary` | yes | string | concise ground-truth answer (used for faithfulness scoring) |
| `source_hint` | yes | list | VEP doc page names the grader expects to be cited |
| `source_urls` | yes | list | canonical source URLs |
| `must_not_contain` | no | list | strings that indicate a wrong answer (e.g. suggesting `bcftools view` when asked about `filter_vep`) |
| `requires_rest_api` | no | bool | true if the question is REST-specific |
| `real_world_from` | no | object | optional provenance: type (`github_issue`, `pattern`), ref, pattern |
| `tags` | no | list | free-form tags |
| `evaluation_notes` | no | string | grader notes |

## How it is used

1. **Retrieval eval**: feed the `question` string into the ingest+search stack, measure whether the retrieved chunks cite any `source_hint`
2. **Faithfulness eval**: run the full RAG chain, then use RAGAS or a deterministic lexical check against `expected_answer_keywords`
3. **Regression tracking**: every prototype iteration runs the full 40-question set; a CSV diff between runs flags regressions

## Sourcing principles

- Every category reflects a real class of user question seen in public support threads
- No synthetic questions that do not have a plausible user behind them
- `must_not_contain` deliberately captures common wrong answers, not just any wrong answer, so the benchmark scores correlate with real support quality

## Data provenance statement

Questions are not verbatim copies of any single public issue. They are synthesised patterns from browsing issue titles and helpdesk FAQ structure. If any question closely resembles a specific public issue and the author wants it removed, open an issue on the prototype repo and we remove it

## Next steps (deferred to later commits)

- add 20 more questions to reach n=60 once we have eval numbers on the first 40
- stratified split (train 32 / held-out 8) for prompt tuning
- per-question RAGAS ground-truth in a separate file (`golden_answers.jsonl`) so humans can edit one without touching the other
