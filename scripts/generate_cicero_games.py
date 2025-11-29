#!/usr/bin/env python
"""
Generate full Diplomacy games with Cicero playing all powers and sample random states.

This script:
1. Loads the Cicero (BQRE1P) agent
2. Runs complete games where Cicero plays all 7 powers
3. Captures data at each phase during gameplay:
   - Orders for all powers
   - Policy probabilities (action distributions)
   - Value estimates (expected payoffs)

Usage:
    python scripts/generate_cicero_games.py --num-games 10 --samples-per-game 10

Prerequisites:
    - pydipcc must be built (run 'make' to compile C++ extension)
    - Model files must exist (run 'bin/download_model_files.sh')
    - GPU recommended (Cicero is slow on CPU)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the diplomacy_cicero submodule is in path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DIPLOMACY_ROOT = PROJECT_ROOT / "diplomacy_cicero"
sys.path.insert(0, str(DIPLOMACY_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for pydipcc availability
try:
    from fairdiplomacy import pydipcc
    from fairdiplomacy.agents import build_agent_from_cfg
    from fairdiplomacy.agents.base_agent import BaseAgent
    from fairdiplomacy.agents.base_search_agent import BaseSearchAgent, SearchResult
    from fairdiplomacy.agents.player import Player
    from fairdiplomacy.models.consts import POWERS
    from fairdiplomacy.utils.game import game_from_view_of
    from fairdiplomacy.typedefs import Power, Action, PowerPolicies
    import heyhi
    import conf.agents_cfgs
    PYDIPCC_AVAILABLE = True
except ImportError as e:
    PYDIPCC_AVAILABLE = False
    IMPORT_ERROR = str(e)


def load_cicero_agent(cfg_path: str):
    """
    Load the Cicero agent from config file.

    Args:
        cfg_path: Path to agent config (e.g., 'conf/common/agents/cicero.prototxt')

    Returns:
        Loaded Cicero agent instance
    """
    logger.info(f"Loading Cicero agent from {cfg_path}...")
    cfg = heyhi.load_config(cfg_path, msg_class=conf.agents_cfgs.Agent)
    agent = build_agent_from_cfg(cfg)
    logger.info("Cicero agent loaded successfully")
    return agent


def policy_to_serializable(policy: PowerPolicies, top_k: int = 10) -> Dict[str, List[Dict]]:
    """
    Convert PowerPolicies to a JSON-serializable format.
    Only keeps top_k actions per power to limit size.

    Args:
        policy: Dict[Power, Dict[Action, float]]
        top_k: Number of top actions to keep per power

    Returns:
        Dict with power -> list of {action, prob} dicts
    """
    result = {}
    for power, action_probs in policy.items():
        # Sort by probability descending
        sorted_actions = sorted(action_probs.items(), key=lambda x: -x[1])[:top_k]
        result[power] = [
            {
                'action': list(action) if isinstance(action, tuple) else action,
                'prob': float(prob)
            }
            for action, prob in sorted_actions
        ]
    return result


def run_full_game_with_rich_data(
    agent: BaseAgent,
    game_id: str,
    max_year: int = 1920,
    seed: int = 0,
    capture_policies: bool = True,
    capture_values: bool = True,
    policy_top_k: int = 10,
) -> Tuple[pydipcc.Game, List[Dict]]:
    """
    Run a complete game with Cicero playing all 7 powers.
    Captures data at each phase including orders, policies, and values.

    Args:
        agent: The Cicero agent instance
        game_id: Unique identifier for this game
        max_year: Stop game after this year
        seed: Random seed for reproducibility
        capture_policies: Whether to capture action probability distributions
        capture_values: Whether to capture value estimates
        policy_top_k: Number of top actions to keep in policy output

    Returns:
        Tuple of (final game object, list of phase snapshots)
    """
    logger.info(f"Starting game {game_id} (max_year={max_year}, seed={seed})")

    # Set seeds
    random.seed(seed)

    game = pydipcc.Game()

    # Create Player wrappers for each power (needed to access run_search)
    players: Dict[Power, Player] = {
        power: Player(agent, power) for power in POWERS
    }

    # Check if agent supports search (for policies/values)
    is_search_agent = isinstance(agent, BaseSearchAgent)

    # Collect snapshots with rich data as we play
    phase_snapshots = []
    turn_count = 0

    # Game loop - get orders and capture state at each turn
    while not game.is_game_done:
        phase = game.current_short_phase

        # Check max year
        year = int(phase[1:5])
        if year > max_year:
            logger.info(f"Reached max year {max_year}, stopping game {game_id}")
            break

        logger.debug(f"Processing phase {phase}")

        # Extract game state features (before orders)
        units = game.get_units()
        centers = game.get_state()["centers"]

        # Get orders and optionally run search for each power
        power_orders: Dict[Power, Action] = {}
        power_policies: Dict[Power, Any] = {}
        power_values: Dict[Power, float] = {}
        power_action_values: Dict[Power, Dict[str, float]] = {}

        for power in POWERS:
            orderable_locs = game.get_orderable_locations().get(power, [])

            if not orderable_locs:
                # No orders needed for this power
                power_orders[power] = ()
                continue

            # Get game from this power's view
            game_view = game_from_view_of(game, power)

            if is_search_agent and (capture_policies or capture_values):
                # Run search to get policies and values
                try:
                    search_result: SearchResult = players[power].run_search(
                        game_view, allow_early_exit=False
                    )

                    # Sample action from search result
                    orders = search_result.sample_action(power)
                    power_orders[power] = orders

                    # Capture policy (probability distribution over actions)
                    if capture_policies:
                        try:
                            agent_policy = search_result.get_agent_policy()
                            if power in agent_policy:
                                power_policies[power] = policy_to_serializable(
                                    {power: agent_policy[power]}, top_k=policy_top_k
                                )[power]
                        except Exception as e:
                            logger.debug(f"Could not get policy for {power}: {e}")

                    # Capture value estimates
                    if capture_values:
                        try:
                            power_values[power] = float(search_result.avg_utility(power))
                            # Also get value of the selected action
                            power_action_values[power] = {
                                'selected_action_value': float(
                                    search_result.avg_action_utility(power, orders)
                                )
                            }
                        except Exception as e:
                            logger.debug(f"Could not get values for {power}: {e}")

                except Exception as e:
                    logger.warning(f"Search failed for {power} at {phase}: {e}")
                    # Fall back to direct order query
                    orders = players[power].get_orders(game_view)
                    power_orders[power] = orders
            else:
                # Non-search agent or not capturing extra data
                orders = players[power].get_orders(game_view)
                power_orders[power] = orders

        # Create snapshot for each power that has orderable locations
        for power in POWERS:
            orderable_locs = game.get_orderable_locations().get(power, [])
            if not orderable_locs:
                continue
            orders = power_orders.get(power, ())
            orders = list(orders) if isinstance(orders, tuple) else orders

            snapshot = {
                'game_id': game_id,
                'phase_name': phase,
                'focal_power': power,
                'game_json': game.to_json(),
                'units': list(units.get(power, [])),
                'centers': list(centers.get(power, [])),
                'orderable_locations': list(orderable_locs),
                'all_units': {p: list(u) for p, u in units.items()},
                'all_centers': {p: list(c) for p, c in centers.items()},
                'orders': orders,
                'num_orders': len(orders),
            }

            # Add policy if captured
            if power in power_policies:
                snapshot['policy'] = power_policies[power]

            # Add values if captured
            if power in power_values:
                snapshot['value'] = power_values[power]
            if power in power_action_values:
                snapshot['action_value'] = power_action_values[power]['selected_action_value']

            phase_snapshots.append(snapshot)

        # Set orders and process
        for power, orders in power_orders.items():
            if game.get_orderable_locations().get(power):
                game.set_orders(power, list(orders))

        game.process()
        turn_count += 1

        if turn_count % 5 == 0:
            logger.info(f"Game {game_id}: processed {turn_count} turns, now at {game.current_short_phase}")

    logger.info(f"Game {game_id} completed: {turn_count} turns, {len(phase_snapshots)} power-phase snapshots")

    return game, phase_snapshots


def sample_states(
    all_snapshots: List[Dict],
    samples_per_game: int = 10,
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Randomly sample states from collected snapshots.

    Args:
        all_snapshots: List of all snapshots from all games
        samples_per_game: Number of states to sample per game
        seed: Random seed for sampling

    Returns:
        List of sampled state dictionaries
    """
    if seed is not None:
        random.seed(seed)

    # Group by game_id
    by_game = {}
    for snap in all_snapshots:
        gid = snap['game_id']
        if gid not in by_game:
            by_game[gid] = []
        by_game[gid].append(snap)

    logger.info(f"Sampling from {len(by_game)} games ({len(all_snapshots)} total snapshots)")

    sampled = []
    for game_id, game_snapshots in by_game.items():
        if not game_snapshots:
            continue

        # Random sampling
        n_samples = min(samples_per_game, len(game_snapshots))
        selected = random.sample(game_snapshots, n_samples)

        # Remove game_json from output (too large)
        for snap in selected:
            sampled.append({k: v for k, v in snap.items() if k != 'game_json'})

    logger.info(f"Sampled {len(sampled)} states total")
    return sampled


def save_game(game: pydipcc.Game, output_path: str) -> None:
    """Save a game to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(game.to_json())
    logger.debug(f"Saved game to {output_path}")


def write_jsonl(data: List[Dict], output_path: str) -> None:
    """Write data to JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for row in data:
            f.write(json.dumps(row) + '\n')
    logger.info(f"Wrote {len(data)} rows to {output_path}")


def compute_statistics(sampled: List[Dict]) -> None:
    """Print statistics about sampled data."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total samples: {len(sampled)}")

    if not sampled:
        return

    # Phase type distribution
    phase_types = {}
    for row in sampled:
        pt = row['phase_name'][-1]
        phase_types[pt] = phase_types.get(pt, 0) + 1
    print(f"Phase types: {phase_types}")

    # Power distribution
    powers = {}
    for row in sampled:
        p = row['focal_power']
        powers[p] = powers.get(p, 0) + 1
    print(f"Powers: {powers}")

    # Year distribution
    years = {}
    for row in sampled:
        y = int(row['phase_name'][1:5])
        years[y] = years.get(y, 0) + 1
    print(f"Years: {dict(sorted(years.items()))}")

    # Orders per sample
    order_counts = [row['num_orders'] for row in sampled]
    avg_orders = sum(order_counts) / len(order_counts)
    print(f"Average orders per sample: {avg_orders:.2f}")
    print(f"Min/Max orders: {min(order_counts)}/{max(order_counts)}")

    # Games represented
    games = set(row['game_id'] for row in sampled)
    print(f"Games represented: {len(games)}")

    # Policy coverage
    with_policy = sum(1 for row in sampled if 'policy' in row)
    print(f"Samples with policy: {with_policy}/{len(sampled)} ({100*with_policy/len(sampled):.1f}%)")

    # Value coverage
    with_value = sum(1 for row in sampled if 'value' in row)
    print(f"Samples with value: {with_value}/{len(sampled)} ({100*with_value/len(sampled):.1f}%)")

    # Value statistics
    if with_value > 0:
        values = [row['value'] for row in sampled if 'value' in row]
        avg_value = sum(values) / len(values)
        print(f"Average value: {avg_value:.4f}")
        print(f"Value range: [{min(values):.4f}, {max(values):.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate full Diplomacy games with Cicero and sample random states"
    )
    parser.add_argument(
        "--num-games", type=int, default=10,
        help="Number of complete games to generate (default: 10)"
    )
    parser.add_argument(
        "--samples-per-game", type=int, default=10,
        help="Number of states to sample from each game (default: 10)"
    )
    parser.add_argument(
        "--max-year", type=int, default=1920,
        help="Maximum game year (default: 1920)"
    )
    parser.add_argument(
        "--agent-cfg", type=str, default="diplomacy_cicero/conf/common/agents/cicero.prototxt",
        help="Path to Cicero agent config file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/cicero_games",
        help="Output directory for games and sampled states"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no-save-games", action="store_true",
        help="Don't save full game JSON files"
    )
    parser.add_argument(
        "--no-capture-policies", action="store_true",
        help="Don't capture action probability distributions"
    )
    parser.add_argument(
        "--no-capture-values", action="store_true",
        help="Don't capture value estimates"
    )
    parser.add_argument(
        "--policy-top-k", type=int, default=10,
        help="Number of top actions to keep in policy output (default: 10)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check prerequisites
    if not PYDIPCC_AVAILABLE:
        print("ERROR: pydipcc is not available.")
        print(f"Import error: {IMPORT_ERROR}")
        print("\nTo fix this:")
        print("  1. Run 'make' to build the C++ extension")
        print("  2. Ensure model files are downloaded (bin/download_model_files.sh)")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Cicero Game Generator")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Number of games: {args.num_games}")
    print(f"Samples per game: {args.samples_per_game}")
    print(f"Max year: {args.max_year}")
    print(f"Agent config: {args.agent_cfg}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Capture policies: {not args.no_capture_policies}")
    print(f"Capture values: {not args.no_capture_values}")
    print(f"Policy top-k: {args.policy_top_k}")
    print("=" * 60 + "\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Cicero agent (expensive - do once)
    agent = load_cicero_agent(args.agent_cfg)

    # Generate games and collect all snapshots
    all_snapshots = []
    for i in range(args.num_games):
        game_id = f"cicero_{i:04d}"
        game, snapshots = run_full_game_with_rich_data(
            agent,
            game_id=game_id,
            max_year=args.max_year,
            seed=args.seed + i,
            capture_policies=not args.no_capture_policies,
            capture_values=not args.no_capture_values,
            policy_top_k=args.policy_top_k,
        )
        all_snapshots.extend(snapshots)

        # Save full game
        if not args.no_save_games:
            game_path = output_dir / f"game_{i:04d}.json"
            save_game(game, str(game_path))

        # Log final scores
        final_centers = game.get_state()["centers"]
        scores = {p: len(c) for p, c in final_centers.items() if c}
        logger.info(f"Game {game_id} final scores: {scores}")

    # Sample random states (orders already included from game generation)
    sampled = sample_states(
        all_snapshots,
        samples_per_game=args.samples_per_game,
        seed=args.seed
    )

    # Write sampled states to JSONL
    output_path = output_dir / "sampled_states.jsonl"
    write_jsonl(sampled, str(output_path))

    # Print statistics
    compute_statistics(sampled)

    print("\n" + "=" * 60)
    print(f"Done! Generated {args.num_games} games")
    print(f"Collected {len(all_snapshots)} total power-phase snapshots")
    print(f"Sampled {len(sampled)} states with Cicero orders")
    print(f"Output: {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
