"""Optionally prepare reusable frozen ESM-2 features before training/prediction."""
import sys
from immunovae.cli import main

if __name__ == '__main__':
    main(['cache', *sys.argv[1:]])
