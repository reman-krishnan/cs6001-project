# Commit Now, Walk Away Later: Designing Forfeits for Delayed Blockchain Transactions

CS 6001 course project by Reman Krishnan (Roll No. 26D0365), IIT Bombay.

**Start here: the final report is [`report/report.pdf`](report/report.pdf).**

## The project in one paragraph
Post-quantum signatures are about forty times larger than today's, so some new blockchain designs keep the signature off the ledger. The user first posts a short commitment and authorizes the transaction a few blocks later. That delay gives the user a free option: watch the price, and walk away if it moves against the trade. The project treats this as a **mechanism design problem**. A designer chooses a forfeit (a penalty for walking away), where its money goes, and whether the block producer may complete the transaction itself. The goal is to deter speculation while keeping honest users willing to participate, keeping the producer willing to publish, making collusion between user and producer unprofitable, and keeping the locked collateral small. The protocol is modelled as an extensive-form game with chance moves and solved for subgame-perfect equilibria by backward induction. Proofs cover the cases with closed forms, and a verified solver calibrated to nine years of ETH prices covers the rest.

## Main results
| Question | Answer |
|---|---|
| Q1. How large must a forfeit be on its own? | 0.39% to 0.74% of the trade for a 20-minute window, rising with window and volatility |
| Q2. Where should the money go? | Out of the user-producer pair. If the forfeit stays entirely inside it, the pair can profit together (Proposition 4, an impossibility result), and burning all of it needs the smallest forfeit (Proposition 7) |
| Q3. When can producers undo the design? | At 20 minutes, a burned forfeit of 1.35% keeps the pair's joint gain over the best equilibrium within 0.1 basis points. The required forfeit shrinks with the window and fits a 1% collateral limit only for windows up to 11.8 minutes |
| Q4. Bond or race for a disputed slot? | A large enough bond deters if the owner can afford it and strictly prefers to answer; a race deters only costly attacks |

## Repository layout
| Folder | What it holds |
|---|---|
| [`report/`](report/) | The report (`report.pdf`), its LaTeX source, and the generated tables and figures in `assets/` |
| [`dagame/`](dagame/) | The Python package: game model, backward-induction solver, game-tree cross-check, coalition and contest models |
| [`scripts/`](scripts/) | Programs that produce every result, table and figure ([`docs/CODE_MAP.md`](docs/CODE_MAP.md) says which does what) |
| [`results/`](results/) | Every result file used in the report; `results/main/` is the main calibration ([`results/README.md`](results/README.md)) |
| [`data/`](data/) | Frozen Binance price data used for calibration ([`data/README.md`](data/README.md)) |
| [`tests/`](tests/) | 196 automated tests, including hand-solved games and deliberately injected bugs that must be caught |
| [`docs/`](docs/) | The project proposal, how to reproduce everything, and a map from each question to its code and output |

## Reproduce
```
git clone https://github.com/reman-krishnan/cs6001-project.git
cd cs6001-project

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 -m pytest -q
./run_all.sh
```
Details, the environment and individual steps are in [`docs/REPRODUCING.md`](docs/REPRODUCING.md).
