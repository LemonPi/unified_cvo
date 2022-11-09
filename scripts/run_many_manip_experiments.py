import argparse
import subprocess
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))


parser = argparse.ArgumentParser(description='Run many CVO alignment experiments')
parser.add_argument('--seed', metavar='N', type=int, nargs='+',
                    default=[0, 1, 2, 3, 4],
                    help='random seed(s) to run')
parser.add_argument('--task', type=str, nargs='+',
                    default=['mustard'],
                    help='what tasks to run')
parser.add_argument('--params', default=f"{ROOT_DIR}/cvo_params/johnson_manip_params.yaml", help='CVO parameters file')
parser.add_argument('--dry', action='store_true', help='print the commands to run without execution')

args = parser.parse_args()

if __name__ == "__main__":
    for task in args.task:
        for seed in args.seed:
            source_file = f"{ROOT_DIR}/data/poke/{task.upper()}_{seed}.txt"
            target_file = f"{ROOT_DIR}/data/poke/{task.upper()}.txt"
            to_run = [f"{ROOT_DIR}/build/bin/cvo_align_manip_freespace", source_file, target_file, args.params]

            print(" ".join(to_run))
            if not args.dry:
                subprocess.run(to_run)
