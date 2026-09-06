# Kwak choice-orientation correction

- **Status:** success (15/15 cells)
- **W&B group:** `gru-kwak-matched-half@20260906-071413`
- **Beaker experiment:** [`01M1VH47Q8M8N75FKM2F5NX7AT`](https://beaker.org/ex/01M1VH47Q8M8N75FKM2F5NX7AT)
- **Corrected data:** `study09-kwak-choicefix-20260906` (`01M1VGY7MV148S9GWM2HW38FRQ`)
- **Supersedes:** `gru-kwak-matched-half@20260905-232924`

The released Kwak choice bit is `0=right, 1=left`; the canonical contract is
`0=left, 1=right`. This rerun uses `animal_response = 1 - source_choice` while
leaving the released left/right reward-probability columns in place. The split
manifest and all trial counts are unchanged.

Three-seed mean held-out normalized likelihood is `0.61842`, `0.61941`,
`0.62036`, `0.62046`, and `0.62029` for D=10, 30, 100, 300, and 614,
respectively. The corrected D=614 value is 0.00036 below the superseded value;
the qualitative scaling curve is unchanged.
