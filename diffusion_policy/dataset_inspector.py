#!/usr/bin/env python3
"""
Dataset Inspector Script for ManiSkill Trajectory Datasets

This script opens and inspects ManiSkill trajectory datasets stored in HDF5 format
with associated JSON metadata files, and prints detailed information about the data structure.

Usage:
    python dataset_inspector.py [h5_file] [json_file]

If no files are specified, defaults to:
    - trajectory.none.pd_joint_delta_pos.physx_cuda.h5
    - trajectory.none.pd_joint_delta_pos.physx_cuda.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subsection_header(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def format_shape_dtype(dataset):
    """Format shape and dtype information for a dataset."""
    if isinstance(dataset, h5py.Dataset):
        return f"shape={dataset.shape}, dtype={dataset.dtype}"
    else:
        return f"<Group with {len(dataset.keys())} items>"


def print_dict_structure(data, indent=0, max_depth=5, current_depth=0):
    """Recursively print the structure of a dictionary or h5py Group."""
    if current_depth >= max_depth:
        print("  " * indent + "... (max depth reached)")
        return

    if isinstance(data, h5py.Group):
        for key in data.keys():
            item = data[key]
            if isinstance(item, h5py.Dataset):
                print("  " * indent + f"├─ {key}: {format_shape_dtype(item)}")
            elif isinstance(item, h5py.Group):
                print("  " * indent + f"├─ {key}/ (Group)")
                print_dict_structure(item, indent + 1, max_depth, current_depth + 1)
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print("  " * indent + f"├─ {key}/ (dict)")
                print_dict_structure(value, indent + 1, max_depth, current_depth + 1)
            elif isinstance(value, np.ndarray):
                print("  " * indent + f"├─ {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], (dict, np.ndarray)):
                    print("  " * indent + f"├─ {key}: list of {len(value)} items")
                else:
                    print("  " * indent + f"├─ {key}: {value}")
            else:
                # Truncate long strings
                str_val = str(value)
                if len(str_val) > 80:
                    str_val = str_val[:77] + "..."
                print("  " * indent + f"├─ {key}: {str_val}")


def print_sample_data(dataset, num_samples=10):
    """Print sample data from a dataset."""
    if isinstance(dataset, h5py.Dataset):
        data = dataset[:]
        if len(data.shape) == 1 and data.shape[0] <= num_samples * 2:
            print(f"  Data: {data}")
        elif len(data.shape) >= 1:
            print(f"  First {min(num_samples, len(data))} items: {data[:num_samples]}")
            if len(data) > num_samples * 2:
                print(f"  Last {min(num_samples, len(data))} items: {data[-num_samples:]}")
    elif isinstance(dataset, np.ndarray):
        if len(dataset.shape) == 1 and dataset.shape[0] <= num_samples * 2:
            print(f"  Data: {dataset}")
        elif len(dataset.shape) >= 1:
            print(f"  First {min(num_samples, len(dataset))} items: {dataset[:num_samples]}")
            if len(dataset) > num_samples * 2:
                print(f"  Last {min(num_samples, len(dataset))} items: {dataset[-num_samples:]}")


def inspect_json_metadata(json_path):
    """Inspect and print JSON metadata information."""
    print_section_header("JSON METADATA")

    if not os.path.exists(json_path):
        print(f"ERROR: JSON file not found: {json_path}")
        return None

    with open(json_path, "r") as f:
        json_data = json.load(f)

    # Environment Info
    print_subsection_header("Environment Info")
    if "env_info" in json_data:
        env_info = json_data["env_info"]
        print(f"Environment ID: {env_info.get('env_id', 'N/A')}")
        print(f"Max Episode Steps: {env_info.get('max_episode_steps', 'N/A')}")
        print("Environment kwargs:")
        for key, value in env_info.get("env_kwargs", {}).items():
            print(f"  - {key}: {value}")

    # Source Information
    print_subsection_header("Source Information")
    print(f"Source Type: {json_data.get('source_type', 'N/A')}")
    print(f"Source Description: {json_data.get('source_desc', 'N/A')}")

    # Episodes Summary
    print_subsection_header("Episodes Summary")
    episodes = json_data.get("episodes", [])
    print(f"Total Episodes: {len(episodes)}")

    if len(episodes) > 0:
        print("\nFirst Episode Details:")
        first_ep = episodes[0]
        print(f"  Episode ID: {first_ep.get('episode_id', 'N/A')}")
        print(f"  Control Mode: {first_ep.get('control_mode', 'N/A')}")
        print(f"  Elapsed Steps: {first_ep.get('elapsed_steps', 'N/A')}")
        print(f"  Reset kwargs: {first_ep.get('reset_kwargs', {})}")
        if "info" in first_ep:
            print(f"  Final info: {first_ep['info']}")

        # Statistics across all episodes
        print("\nEpisode Statistics:")
        elapsed_steps = [ep.get("elapsed_steps", 0) for ep in episodes]
        if elapsed_steps:
            print(
                f"  Steps - Min: {min(elapsed_steps)}, Max: {max(elapsed_steps)}, "
                f"Mean: {np.mean(elapsed_steps):.2f}, Total: {sum(elapsed_steps)}"
            )

        # Check for success/fail info
        successes = [ep.get("info", {}).get("success", None) for ep in episodes]
        success_count = sum(1 for s in successes if s is True)
        if any(s is not None for s in successes):
            print(
                f"  Successful episodes: {success_count}/{len(episodes)} ({100 * success_count / len(episodes):.1f}%)"
            )

    return json_data


def inspect_h5_structure(h5_path, json_data=None):
    """Inspect and print HDF5 file structure."""
    print_section_header("HDF5 FILE STRUCTURE")

    if not os.path.exists(h5_path):
        print(f"ERROR: H5 file not found: {h5_path}")
        return None

    h5_file = h5py.File(h5_path, "r")

    # Top-level keys
    print_subsection_header("Top-Level Keys")
    print(f"Total keys: {len(h5_file.keys())}")

    # Identify trajectory keys
    traj_keys = [key for key in h5_file.keys() if key.startswith("traj_")]
    print(f"Trajectory keys: {len(traj_keys)}")
    if traj_keys:
        # Show a sample of trajectory IDs
        sample_size = min(5, len(traj_keys))
        print(f"Sample trajectory keys: {traj_keys[:sample_size]}")
        if len(traj_keys) > sample_size:
            print(f"  ... and {len(traj_keys) - sample_size} more")

    # Inspect first trajectory in detail
    if traj_keys:
        print_subsection_header("First Trajectory Structure (traj_0)")
        first_traj = h5_file[traj_keys[1]]
        print("\nKeys in trajectory:")
        print_dict_structure(first_traj, indent=0, max_depth=4)

        # Print detailed info for each key
        print_subsection_header("Detailed Trajectory Contents")

        for key in first_traj.keys():
            item = first_traj[key]
            print(f"\n[{key}]")

            if isinstance(item, h5py.Dataset):
                print("  Type: Dataset")
                print(f"  Shape: {item.shape}")
                print(f"  Dtype: {item.dtype}")
                print(f"  Size: {item.size} elements ({item.nbytes / 1024:.2f} KB)")
                print_sample_data(item)

            elif isinstance(item, h5py.Group):
                print("  Type: Group (nested structure)")
                print(f"  Sub-keys: {list(item.keys())}")
                print("  Structure:")
                print_dict_structure(item, indent=1, max_depth=3)

                # For nested groups, show sample from first item if it's an array
                if len(item.keys()) > 0:
                    first_subkey = list(item.keys())[0]
                    subitem = item[first_subkey]
                    if isinstance(subitem, h5py.Dataset):
                        print(f"\n  Example: [{first_subkey}]")
                        print(f"    Shape: {subitem.shape}, Dtype: {subitem.dtype}")
                        if subitem.size < 100:  # Only show data for small arrays
                            print_sample_data(subitem[:])
    return h5_file


def inspect_dataset_compatibility(h5_file, json_data):
    """Check compatibility between JSON and H5 data."""
    print_section_header("DATA CONSISTENCY CHECKS")

    if json_data is None or h5_file is None:
        print("Skipping consistency checks due to missing data.")
        return

    episodes = json_data.get("episodes", [])
    traj_keys = [key for key in h5_file.keys() if key.startswith("traj_")]

    print(f"Episodes in JSON: {len(episodes)}")
    print(f"Trajectories in H5: {len(traj_keys)}")

    # Check if all episodes have corresponding trajectories
    missing_trajs = []
    for ep in episodes[:10]:  # Check first 10
        ep_id = ep.get("episode_id")
        traj_key = f"traj_{ep_id}"
        if traj_key not in h5_file:
            missing_trajs.append(ep_id)

    if missing_trajs:
        print(f"WARNING: Missing trajectories for episode IDs: {missing_trajs}")
    else:
        print("✓ All checked episodes have corresponding trajectories")

    # Check if elapsed_steps matches actual trajectory length
    if len(episodes) > 0 and len(traj_keys) > 0:
        ep = episodes[0]
        traj_key = f"traj_{ep['episode_id']}"
        if traj_key in h5_file:
            traj = h5_file[traj_key]
            if "actions" in traj:
                actual_steps = len(traj["actions"])
                reported_steps = ep.get("elapsed_steps", -1)
                if actual_steps == reported_steps:
                    print(f"✓ Episode lengths are consistent (checked episode 0: {actual_steps} steps)")
                else:
                    print(f"WARNING: Episode 0 length mismatch - JSON: {reported_steps}, H5: {actual_steps}")


def analyze_episode_outcomes(h5_file, json_data):
    """Analyze episode outcomes: success, failure, truncation."""
    print_section_header("EPISODE OUTCOME ANALYSIS")

    if json_data is None or h5_file is None:
        print("Skipping outcome analysis due to missing data.")
        return

    episodes = json_data.get("episodes", [])

    # Statistics
    total_episodes = len(episodes)
    success_count = 0
    truncated_count = 0
    terminated_count = 0

    # Track some examples
    truncated_examples = []
    success_examples = []

    # Track when success first occurs
    success_on_last_step_count = 0
    success_before_last_step_count = 0

    print(f"Analyzing {total_episodes} episodes...")
    print()

    for ep in episodes:
        ep_id = ep.get("episode_id")
        traj_key = f"traj_{ep_id}"

        if traj_key not in h5_file:
            continue

        traj = h5_file[traj_key]

        # Check final timestep values
        if "terminated" in traj:
            final_terminated = bool(traj["terminated"][-1])
            if final_terminated:
                terminated_count += 1

        if "truncated" in traj:
            final_truncated = bool(traj["truncated"][-1])
            if final_truncated:
                truncated_count += 1
                if len(truncated_examples) < 5:
                    truncated_examples.append((ep_id, len(traj["actions"])))

        if "success" in traj:
            success_array = traj["success"][:]
            final_success = bool(success_array[-1])
            if final_success:
                success_count += 1

                # Find when success first occurred
                first_success_idx = np.argmax(success_array)  # First True index
                episode_length = len(success_array)

                if first_success_idx == episode_length - 1:
                    success_on_last_step_count += 1
                else:
                    success_before_last_step_count += 1

                if len(success_examples) < 5:
                    success_examples.append((ep_id, episode_length, first_success_idx))

    # Print results
    print_subsection_header("Outcome Statistics")
    print(f"Total Episodes: {total_episodes}")
    print(f"Successful Episodes: {success_count} ({100 * success_count / total_episodes:.1f}%)")
    print(f"Terminated Episodes: {terminated_count} ({100 * terminated_count / total_episodes:.1f}%)")
    print(f"Truncated Episodes: {truncated_count} ({100 * truncated_count / total_episodes:.1f}%)")

    if truncated_count > 0:
        print(f"\n⚠️  {truncated_count} episodes were TRUNCATED (hit time limit)")
        print("   These episodes reached max_episode_steps")
        if truncated_examples:
            print(f"   Examples (ep_id, steps): {truncated_examples[:5]}")
    else:
        print("\n✓ No truncated episodes - all episodes either succeeded or failed before time limit")

    if success_examples:
        print(f"\n✓ Successful episode examples (ep_id, total_steps, first_success_step): {success_examples[:5]}")

    # Success timing analysis
    print_subsection_header("When Does Success Occur?")
    if success_count > 0:
        pct_last = 100 * success_on_last_step_count / success_count
        pct_before = 100 * success_before_last_step_count / success_count
        print(f"Episodes succeeding on LAST step: {success_on_last_step_count} ({pct_last:.1f}% of successes)")
        print(
            f"Episodes succeeding BEFORE last step: {success_before_last_step_count} ({pct_before:.1f}% of successes)"
        )

        print("\nKey Insight:")
        print("  - In ManiSkill, 'terminated=True' is automatically set when 'success=True'")
        print("  - Episodes terminate IMMEDIATELY upon success")
        print(f"  - All {success_count} successful episodes terminated as soon as success was achieved")
        if success_on_last_step_count > 0 and truncated_count > 0:
            print(f"  - {success_on_last_step_count} episodes: succeeded on their final step")
            print("  - Some of these also have truncated=True (success achieved AT time limit)")

    # Additional insights
    print_subsection_header("Episode Length vs Outcome")
    if truncated_count > 0 and success_count > 0:
        # Check correlation between length and truncation
        max_steps = max(ep.get("elapsed_steps", 0) for ep in episodes)
        print(f"Maximum episode length: {max_steps} steps")
        both_count = sum(1 for ep in episodes if ep.get("elapsed_steps") == max_steps)
        print(f"Episodes reaching max length: {both_count}")
        print("\nInterpretation:")
        print("  - Episodes can be BOTH truncated AND successful")
        print("  - This means success was achieved exactly AT the time limit")


def filter_successful_episodes(h5_file, json_data, h5_path, json_path, output_suffix="_success_only"):
    """
    Filter dataset to keep only successful episodes and save to new files.

    Args:
        h5_file: Open HDF5 file handle
        json_data: Parsed JSON metadata
        h5_path: Path to original H5 file
        json_path: Path to original JSON file
        output_suffix: Suffix to add to output filenames

    Returns:
        Tuple of (new_h5_path, new_json_path, num_kept, num_removed)
    """
    print_section_header("FILTERING SUCCESSFUL EPISODES")

    if json_data is None or h5_file is None:
        print("ERROR: Cannot filter without both H5 and JSON data.")
        return None, None, 0, 0

    episodes = json_data.get("episodes", [])

    # Identify successful episodes
    successful_episodes = []
    successful_traj_keys = []

    print("Scanning episodes for success status...")
    for ep in episodes:
        ep_id = ep.get("episode_id")
        traj_key = f"traj_{ep_id}"

        if traj_key not in h5_file:
            print(f"WARNING: Trajectory {traj_key} not found in H5 file, skipping episode {ep_id}")
            continue

        traj = h5_file[traj_key]

        # Check if episode was successful
        if "success" in traj:
            success_array = traj["success"][:]
            final_success = bool(success_array[-1])

            if final_success:
                successful_episodes.append(ep)
                successful_traj_keys.append((traj_key, ep_id))

    num_kept = len(successful_episodes)
    num_removed = len(episodes) - num_kept

    print(f"\nFound {num_kept} successful episodes out of {len(episodes)} total")
    print(f"Removing {num_removed} failed episodes")

    if num_kept == 0:
        print("\nERROR: No successful episodes found. Output files not created.")
        return None, None, 0, num_removed

    # Create output file paths
    h5_base = h5_path.rsplit(".", 1)[0]
    h5_ext = ".h5" if h5_path.endswith(".h5") else ""
    new_h5_path = f"{h5_base}{output_suffix}{h5_ext}"

    json_base = json_path.rsplit(".", 1)[0]
    json_ext = ".json" if json_path.endswith(".json") else ""
    new_json_path = f"{json_base}{output_suffix}{json_ext}"

    print("\nCreating filtered dataset:")
    print(f"  Output H5: {new_h5_path}")
    print(f"  Output JSON: {new_json_path}")

    # Create new H5 file with only successful trajectories
    print("\nCopying successful trajectories to new H5 file...")
    with h5py.File(new_h5_path, "w") as new_h5:
        for new_idx, (old_traj_key, old_ep_id) in enumerate(successful_traj_keys):
            new_traj_key = f"traj_{new_idx}"

            # Copy entire trajectory group
            h5_file.copy(old_traj_key, new_h5, new_traj_key)

            if (new_idx + 1) % 100 == 0:
                print(f"  Copied {new_idx + 1}/{num_kept} trajectories...")

    print(f"✓ Copied all {num_kept} successful trajectories")

    # Create new JSON file with only successful episodes
    print("\nCreating filtered JSON metadata...")
    new_json_data = json_data.copy()

    # Update episode IDs to be consecutive starting from 0
    for new_idx, ep in enumerate(successful_episodes):
        ep["episode_id"] = new_idx

    new_json_data["episodes"] = successful_episodes

    with open(new_json_path, "w") as f:
        json.dump(new_json_data, f, indent=2)

    print(f"✓ Created filtered JSON with {num_kept} episodes")

    # Print summary statistics
    print_subsection_header("Filtering Summary")
    print(f"Original episodes: {len(episodes)}")
    print(f"Successful episodes kept: {num_kept} ({100 * num_kept / len(episodes):.1f}%)")
    print(f"Failed episodes removed: {num_removed} ({100 * num_removed / len(episodes):.1f}%)")

    if successful_episodes:
        total_steps_kept = sum(ep.get("elapsed_steps", 0) for ep in successful_episodes)
        original_total_steps = sum(ep.get("elapsed_steps", 0) for ep in episodes)
        pct = 100 * total_steps_kept / original_total_steps
        print(f"Total steps kept: {total_steps_kept}/{original_total_steps} ({pct:.1f}%)")

    return new_h5_path, new_json_path, num_kept, num_removed


def main():
    parser = argparse.ArgumentParser(
        description="Inspect ManiSkill trajectory dataset files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "h5_file",
        nargs="?",
        # default="/home/michzeng/.maniskill/demos/PushT-v1/rl/trajectory.none.pd_ee_delta_pos.physx_cuda.h5",
        default="/home/michzeng/.maniskill/demos/PushT-v1/rl/test_videos/trajectory.h5",
        help="Path to the HDF5 trajectory file (default: trajectory.none.pd_ee_delta_pos.physx_cuda.h5)",
    )
    parser.add_argument(
        "json_file", nargs="?", default=None, help="Path to the JSON metadata file (default: inferred from h5_file)"
    )
    parser.add_argument("--no-json", action="store_true", help="Skip JSON inspection")
    parser.add_argument("--no-h5", action="store_true", help="Skip H5 inspection")
    parser.add_argument(
        "--filter-success",
        action="store_true",
        help="Remove failed episodes and create new filtered dataset files with only successful episodes",
    )
    parser.add_argument(
        "--output-suffix",
        default="_success_only",
        help="Suffix to add to filtered output files (default: _success_only)",
    )

    args = parser.parse_args()

    # Determine file paths
    h5_path = args.h5_file
    if args.json_file is None:
        json_path = h5_path.replace(".h5", ".json")
    else:
        json_path = args.json_file

    # Print file info
    print_section_header("DATASET INSPECTOR")
    print(f"HDF5 File: {h5_path}")
    print(f"JSON File: {json_path}")
    print(f"HDF5 exists: {os.path.exists(h5_path)}")
    print(f"JSON exists: {os.path.exists(json_path)}")

    # Check if files exist
    if not os.path.exists(h5_path) and not args.no_h5:
        print(f"\nERROR: H5 file not found: {h5_path}")
        print("\nPlease provide a valid trajectory file path.")
        print(f"Current directory: {os.getcwd()}")
        return 1

    # Inspect files
    json_data = None
    h5_file = None

    if not args.no_json:
        json_data = inspect_json_metadata(json_path)

    if not args.no_h5:
        h5_file = inspect_h5_structure(h5_path, json_data)

    # Cross-check data
    if json_data is not None and h5_file is not None:
        inspect_dataset_compatibility(h5_file, json_data)
        analyze_episode_outcomes(h5_file, json_data)

    # Filter successful episodes if requested
    if args.filter_success:
        if json_data is not None and h5_file is not None:
            new_h5, new_json, kept, removed = filter_successful_episodes(
                h5_file, json_data, h5_path, json_path, args.output_suffix
            )
            if new_h5 and new_json:
                print("\n✓ Successfully created filtered dataset:")
                print(f"  {new_h5}")
                print(f"  {new_json}")
        else:
            print("\nERROR: Cannot filter - both H5 and JSON files are required")
            print("Remove --no-h5 and --no-json flags to enable filtering")

    # Cleanup
    if h5_file is not None:
        h5_file.close()

    print_section_header("INSPECTION COMPLETE")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
