## ❌ LLM eval: **FAIL**

**31 regression(s), 4 improvement(s); pass rate 78% -> 33% (-45.0 pts)**

Prompt `v1.2.0` · model `gpt-4o-mini` · dataset `v1.0.0` · provider `mock` · run `20261024-183608-v1.2.0-a949`

| Metric | Baseline | This run | Δ |
|---|---|---|---|
| Pass rate | 78.3% | 33.3% | -45.0 pts |
| Category accuracy | 86.7% | 36.7% | -50.0 pts |
| Avg summary score (1-5) | 4.43 | 4.43 | +0.00 |
| Avg latency | 974 ms | 770 ms | -204 ms |
| Total tokens | 36,505 | 6,123 | -30,382 |

**Why:**
- CRITICAL: pass rate dropped 45.0 pts (critical threshold 8.0).
- WARNING: 'account' accuracy dropped 56.2 pts.
- WARNING: 'billing' accuracy dropped 81.2 pts.
- WARNING: 'technical' accuracy dropped 75.0 pts.
- 31 case(s) flipped pass->fail, 4 fail->pass (McNemar p=0.000, likely real).
- Need 7 runs for drift detection (have 3).

<details><summary>Regressed cases</summary>

| Case | Difficulty | Expected | Was | Now |
|---|---|---|---|---|
| `bil-001` | easy | billing | billing | general |
| `bil-002` | easy | billing | billing | general |
| `bil-004` | medium | billing | billing | general |
| `bil-005` | easy | billing | billing | general |
| `bil-007` | hard | billing | billing | general |
| `bil-008` | medium | billing | billing | general |
| `bil-010` | hard | billing | billing | general |
| `bil-011` | medium | billing | billing | general |
| `bil-012` | medium | billing | billing | general |
| `bil-013` | hard | billing | billing | general |
| `bil-014` | hard | billing | billing | general |
| `bil-016` | medium | billing | billing | general |
| `tec-001` | easy | technical | technical | general |
| `tec-003` | easy | technical | technical | general |
| `tec-004` | easy | technical | technical | general |
| `tec-005` | medium | technical | technical | general |
| `tec-006` | medium | technical | technical | general |
| `tec-007` | medium | technical | technical | general |
| `tec-008` | hard | technical | technical | general |
| `tec-009` | easy | technical | technical | general |
| `tec-010` | hard | technical | technical | general |
| `tec-011` | hard | technical | billing | general |
| `tec-012` | hard | technical | technical | general |
| `tec-016` | easy | technical | technical | general |
| `acc-001` | easy | account | account | general |

</details>

> Merge is blocked until the critical regression is fixed or the baseline is deliberately updated.