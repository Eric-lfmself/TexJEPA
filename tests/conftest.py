"""Keep all checks deterministic, tiny and explicitly CPU-bound."""
import torch

def pytest_sessionstart(session):
    torch.set_num_threads(1)
    torch.manual_seed(42)
