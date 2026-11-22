"""Compare every results/*.json with a backup copy, value by value, ignoring only training-time fields.

Usage (from the repository root):
    python tools/compare_results.py [--new results] [--old ../results_backup_before_final_run]

Exit code: 0 if every compared value is identical, 1 if any value differs or a file is missing.
"""
import argparse
import json
import sys
from pathlib import Path

TIME_KEYS = {"train_seconds", "seconds_per_epoch"}
ACCURACY_MARKERS = ("acc", "gap", "confusion")


def is_labelled_run(item):
    """True for an entry of a 'runs' list, which is identified by its configuration name and seed."""
    return isinstance(item, dict) and "seed" in item and isinstance(item.get("config"), dict)


def leaves(node, path=()):
    """Yield (path, value) for every scalar, skipping training-time fields."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key not in TIME_KEYS:
                yield from leaves(value, path + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            tag = f"{value['config']['name']}|seed={value['seed']}" if is_labelled_run(value) else str(index)
            yield from leaves(value, path + (tag,))
    else:
        yield path, node


def total_training_seconds(data):
    """Total recorded training time of a results file."""
    if "runs" in data:
        return sum(run["train_seconds"] for run in data["runs"])
    return data.get("train_seconds")


def is_accuracy_type(path):
    """True for accuracies, accuracy gaps and confusion-matrix counts."""
    return any(marker in part.lower() for part in path for marker in ACCURACY_MARKERS)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new", default="results", type=Path)
    parser.add_argument("--old", default="../results_backup_before_final_run", type=Path)
    args = parser.parse_args()

    new_files = {p.name for p in args.new.glob("*.json")}
    old_files = {p.name for p in args.old.glob("*.json")}
    missing = sorted(new_files ^ old_files)
    compared, identical, differences, times = 0, 0, [], []

    for name in sorted(new_files & old_files):
        new = json.loads((args.new / name).read_text(encoding="utf-8"))
        old = json.loads((args.old / name).read_text(encoding="utf-8"))
        new_values, old_values = dict(leaves(new)), dict(leaves(old))
        for path in sorted(new_values.keys() | old_values.keys()):
            compared += 1
            a, b = old_values.get(path, "<missing>"), new_values.get(path, "<missing>")
            if a == b and type(a) is type(b):
                identical += 1
            else:
                numeric = isinstance(a, (int, float)) and isinstance(b, (int, float))
                differences.append((name, ".".join(path), a, b, abs(a - b) if numeric else "n/a", is_accuracy_type(path)))
        times.append((name, total_training_seconds(old), total_training_seconds(new)))

    print(f"{compared} values compared, {identical} identical, {len(differences)} different")
    if missing:
        print("Files present on only one side:", ", ".join(missing))
    if differences:
        print("\nfile | metric | old | new | abs difference | accuracy-type")
        for row in differences:
            print(" | ".join(str(value) for value in row))
    print(f"\nAccuracy-type differences: {sum(row[5] for row in differences)}")
    print("\nTraining time recorded in each file (seconds; expected to vary)")
    for name, old_time, new_time in times:
        print(f"{name}: old {old_time:.1f}, new {new_time:.1f}")
    return 1 if differences or missing else 0


if __name__ == "__main__":
    sys.exit(main())
