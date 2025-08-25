#!/usr/bin/env python3
import runpy, sys, pathlib
if __name__ == "__main__":
    target = pathlib.Path(__file__).with_name("titanic_moonshot.py")
    sys.exit("Error: titanic_moonshot.py not found") if not target.exists() else runpy.run_path(str(target), run_name="__main__")
