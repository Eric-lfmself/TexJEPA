"""Convenient bounded entry point; accepts the same explicit --dry-run flag."""
from .run_all import main, run_pipeline

__all__ = ["main", "run_pipeline"]

if __name__ == "__main__":
    main()
