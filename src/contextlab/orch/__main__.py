"""Allow `python -m contextlab.orch run ...`."""
from contextlab.orch.run import main

if __name__ == "__main__":
    import sys
    sys.exit(main())