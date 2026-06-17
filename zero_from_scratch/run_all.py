import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4, 5], default=1)
    args = parser.parse_args()

    scripts = {
        1: BASE / "part1_communicate.py",
        2: BASE / "part2_zero1.py",
        3: BASE / "part3_zero3.py",
        4: BASE / "part4_mini_deepspeed.py",
        5: BASE / "part5_multiprocess.py",
    }

    script_path = scripts[args.part]
    print(f"\n{'='*60}")
    print(f"运行 Part {args.part}: {script_path.name}")
    print(f"{'='*60}\n")

    subprocess.run([sys.executable, str(script_path)], check=True)

if __name__ == "__main__":
    main()