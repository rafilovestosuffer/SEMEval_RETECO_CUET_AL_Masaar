# RECOR — Track 2's source benchmark (skim only, per §8.2)

**Ali, Abdallah, Agarwal, Patel, Jatowt** (University of Innsbruck + Oracle AI). *RECOR:
Reasoning-focused Multi-turn Conversational Retrieval Benchmark.* arXiv **2601.05461v1**, 9 Jan 2026.
Repo <https://github.com/RECOR-Benchmark/RECOR>. arXiv id **verified** 16 Sept 2026.

707 conversations / 2,971 turns / 11 domains / 507,141 documents (2,900 positive, 504,241 hard negatives —
a 174:1 ratio). Built by a Decomposition-and-Verification pipeline that turns complex single-turn queries
into fact-grounded dialogues, with atomic facts verified against sources. 487 conversations derive from
BRIGHT, 220 from newly scraped Stack Exchange.

Track 2 is **out of scope** (CLAUDE.md §7) unless the 1 Dec 2026 conditions hold, so this is context only.

## The one result that matters to us

**Metric confirmation.** "We compute nDCG@10 per turn, an average within each domain, then
**macro-average across all 11 domains to ensure equal weight regardless of domain size**." This is
independent confirmation, from the other source benchmark by the same group, of the per-domain macro
averaging we established for Track 1 — the two papers use identical aggregation.

## Numbers, if 2a is ever entered

nDCG@10 by query-construction strategy, macro over 11 domains:

| | Base | Query Rewrite | Reasoning | History | History+Reasoning |
|---|---:|---:|---:|---:|---:|
| Average (8 retrievers) | .236 | .322 | .393 | .440 | **.479** |
| DIVER | .347 | .430 | .496 | .545 | **.584** |
| BM25 | .185 | .288 | .360 | .446 | .489 |

History alone is worth +86%, reasoning alone +67%, together +103% — sub-additive, so they overlap.
BM25 benefits *most* from history (+141%), since lexical matching simply needs more query terms.
RETECO's published 2a baselines (0.1837 turn-only / 0.4539 +history on train) line up with the Base and
History columns here.

Note DIVER leads here too (.584), consistent with TEMPO — reinforcing it as our model of choice.

## Transferable observations

- Counterintuitive: **low-complexity conversations benefit most** from History+Reasoning (+90%) and
  high-complexity least (+59%), because complex queries already carry enough signal. Context augmentation
  helps least where the query is already information-rich — the same principle behind ReasonIR's
  query-length scaling.
- Retrieval→generation correlation is only **r = .42**. Better retrieval does not reliably mean better
  answers, echoing TEMPO's RAG finding. Worth citing as a limitation.
- Limitations stated: English only; no maths/commonsense domains; requires existing QA pairs with
  supporting documents.
