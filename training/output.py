"""Exclusive, non-destructive output ownership for standalone training jobs."""
from contextlib import contextmanager
import os
import fcntl
from pathlib import Path
import re
import shutil
import tempfile
import torch


@contextmanager
def process_file_lock(path):
    """POSIX advisory lock; kernel releases it even on OOM/SIGKILL.

    Keep the inode after unlocking so a third process cannot replace a lock
    while another process already waits/holds it. Empty lock files are benign.
    """
    fd=os.open(path,os.O_CREAT|os.O_WRONLY|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


@contextmanager
def exclusive_training_output(folder, *, continuing=False):
    folder=Path(folder)
    root=Path(__file__).resolve().parents[1]
    if folder.resolve() in (root,*root.parents):
        raise ValueError('Training requires a dedicated output directory')
    if folder.is_symlink() or any(p.is_symlink() for p in folder.parents):
        raise ValueError('Training output cannot follow symlinks')
    folder.parent.mkdir(parents=True,exist_ok=True)
    lock=folder.parent/('.'+folder.name+'.training.lock')
    with process_file_lock(lock):
        if folder.exists():
            if not folder.is_dir(): raise ValueError('Training output must be a directory')
            if not continuing and any(folder.iterdir()):
                raise FileExistsError('Training output is nonempty; use resume or a new output directory')
            if any(path.is_symlink() for path in folder.rglob('*')):
                raise ValueError('Training output descendants cannot be symlinks')
        yield


def resume_checkpoint(folder):
    """Recover an orphan complete epoch only when the latest pointer is absent."""
    folder=Path(folder);latest=folder/'latest.pt'
    if latest.is_file(): return latest
    candidates=sorted((p for p in folder.glob('epoch_*.pt') if re.fullmatch(r'epoch_[0-9]+\.pt',p.name)),
                      key=lambda p:int(p.stem.split('_')[1]),reverse=True)
    if not candidates: raise FileNotFoundError('Resume requires latest.pt or a complete epoch checkpoint')
    path=candidates[0]
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    epoch=int(path.stem.split('_')[1])
    if payload.get('metadata',{}).get('epoch')!=epoch or payload.get('training_state_version')!=1 or payload.get('epoch')!=epoch or payload.get('next_epoch')!=epoch:
        raise ValueError('Orphan checkpoint lacks consistent complete-epoch training state')
    return path


def restore_latest_pointer(folder,checkpoint):
    """Called only after strict resume loading succeeds; preserve all epoch files."""
    latest=Path(folder)/'latest.pt'
    if latest.exists(): return latest
    fd,temp=tempfile.mkstemp(prefix='.latest-',dir=folder);os.close(fd)
    try:
        shutil.copyfile(checkpoint,temp)
        os.replace(temp,latest)
    finally:
        Path(temp).unlink(missing_ok=True)
    return latest
