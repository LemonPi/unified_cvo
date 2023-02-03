import argparse
import subprocess
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

parser = argparse.ArgumentParser(description='Run many CVO alignment experiments')
parser.add_argument('--seed', metavar='N', type=int, nargs='+',
                    default=[0, 1, 2, 3, 4],
                    help='random seed(s) to run')
tasks = [
    'mustard',
    'mustard_sideways',
    'mustard_fallen',
    'mustard_fallen_sideways',
    'drill',
    'drill_opposite',
    'drill_slanted',
    'drill_fallen',
    'hammer',
    'hammer_straight',
    'hammer_fallen',
    'box',
    'box_fallen',
    'can',
    'can_fallen',
    'clamp',
    'clamp_sideways',
]
parser.add_argument('--task', type=str, nargs='+',
                    default=['all'],
                    choices=['all'] + tasks, help='what tasks to run')
parser.add_argument('--params', default=f"{ROOT_DIR}/cvo_params/johnson_manip_params.yaml", help='CVO parameters file')
parser.add_argument('--dry', action='store_true', help='print the commands to run without execution')
parser.add_argument('--real', action='store_true', help='whether to run the real experiments or sim experiments')

args = parser.parse_args()

runs = {}
if __name__ == "__main__":
    if 'all' in args.task:
        args.task = tasks
    for task in args.task:
        for seed in args.seed:
            experiment_name = "poke_real_processed" if args.real else "poke"
            source_file = f"{ROOT_DIR}/data/{experiment_name}/{task.upper()}_{seed}.txt"
            target_file = f"{ROOT_DIR}/data/{experiment_name}/{task.upper()}.txt"
            to_run = [f"{ROOT_DIR}/build/bin/cvo_align_manip_freespace", source_file, target_file, args.params]

            cmd = " ".join(to_run)
            print(cmd)
            if not args.dry:
                completed = subprocess.run(to_run)
                runs[cmd] = completed.returncode

print("\n\n\n")
for cmd, status in runs.items():
    print(f"{'FAILED' if status != 0 else 'SUCCESS'}:\n{cmd}")
print(f"{len(runs)} command run, {len([status for status in runs.values() if status != 0])} failures\n\n\n")
