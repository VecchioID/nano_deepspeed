import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4, 5], default=1)
    args = parser.parse_args()

    scripts = {
        1: "part1_communicate.py",
        2: "part2_zero1.py",
        3: "part3_zero3.py",
        4: "part4_mini_deepspeed.py",
        5: "part5_multiprocess.py",
    }

    print(f"\n{'='*60}")
    print(f"运行 Part {args.part}: {scripts[args.part]}")
    print(f"{'='*60}\n")

    # Run the script using exec
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        f"part{args.part}",
        scripts[args.part]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"part{args.part}"] = mod
    spec.loader.exec_module(mod)

if __name__ == "__main__":
    main()
