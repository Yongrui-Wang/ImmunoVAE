"""Run EL training or IM transfer, preparing ESM-2 features automatically."""
import sys
from immunovae.cli import main

if __name__ == '__main__':
    main(['train', *sys.argv[1:]])
