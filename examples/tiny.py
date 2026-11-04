"""Convenience entrypoint; equivalent to `python -m attention_lab tiny`."""

import sys

from attention_lab.__main__ import main

if __name__ == "__main__":
    sys.argv.insert(1, "tiny")
    main()
