import sys

from .cli import main

if __name__ == "__main__":
    # By default, running 'python -m transcriber' triggers the CLI.
    # To run the GUI, use 'transcriber-gui' or 'python -m transcriber.gui'
    sys.exit(main())
