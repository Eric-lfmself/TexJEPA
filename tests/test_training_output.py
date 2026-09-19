import subprocess
import sys
from pathlib import Path
import pytest
from training.output import exclusive_training_output


def test_active_writer_is_exclusive_and_crash_releases_lock(tmp_path):
    folder=tmp_path/'run'
    with exclusive_training_output(folder):
        with pytest.raises(BlockingIOError):
            with exclusive_training_output(folder): pass
    code="from training.output import exclusive_training_output; import os; from pathlib import Path\nwith exclusive_training_output(Path(__import__('sys').argv[1])): os._exit(0)"
    subprocess.run([sys.executable,'-c',code,str(folder)],check=True,timeout=20,
                   cwd=Path(__file__).resolve().parents[1])
    with exclusive_training_output(folder,continuing=True): pass
