# Diplomacy Training Data

Generate synthetic Diplomacy games using Meta's Cicero agent for training data.

## Setup

```bash
git clone --recursive https://github.com/ivanj-0/diplomacy-training-data
```

Follow the installation instructions in diplomacy_cicero/README.md to set up the environment and download model files.

## Usage

```bash
python scripts/generate_cicero_games.py --num-games 10 --samples-per-game 10
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-games` | 10 | Number of games to generate |
| `--samples-per-game` | 10 | States to sample per game |
| `--max-year` | 1920 | Stop games after this year |
| `--output-dir` | data/cicero_games | Output directory |
| `--seed` | 42 | Random seed |

## Output

- `data/cicero_games/game_XXXX.json` - Full game states
- `data/cicero_games/sampled_states.jsonl` - Sampled training data with orders, policies, and values
