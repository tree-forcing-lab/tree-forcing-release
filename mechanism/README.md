# Mechanism experiments (Section 5 and the appendix "Mechanism Experiments")

| Folder | Experiment |
|---|---|
| `A_hint/` | Hint test: the first true steps of a GSM8K solution are written into the prompt |
| `B_khop/` | Serial steps in one forward pass: synthetic arithmetic chains (k hops) and the GSM8K cost of chain training |

`analysis/prepare_mechanism_inputs.sh` unpacks what `analysis/mechanism_check.py` and `analysis/mechanism_figure.py`
read; `bash reproduce.sh` runs both. Code is under Apache 2.0, data under CC BY 4.0, like the rest of the repository.
