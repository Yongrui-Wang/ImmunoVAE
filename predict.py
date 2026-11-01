"""Run ImmunoVAE prediction, preparing ESM-2 features automatically."""
import sys
from immunovae.cli import main

if __name__ == '__main__':
    main(['predict', *sys.argv[1:]])
