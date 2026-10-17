"""Create an API user and print their key: python scripts/create_api_key.py alice"""

import sys

from ilovecsv.cli import main

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "demo"
    raise SystemExit(main(["create-key", name]))
