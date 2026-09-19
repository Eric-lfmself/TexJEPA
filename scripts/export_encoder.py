"""Export an explicit encoder from a native full checkpoint without rebuilding it."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import tempfile

import torch
from models.checkpoints import _primitive, CheckpointError, checkpoint_sha256


def export_encoder(checkpoint, output, *, component):
    if component not in ('context_encoder','target_encoder','encoder'):
        raise ValueError('Choose context_encoder/target_encoder for I-JEPA or encoder for MAE')
    source=Path(checkpoint).resolve(); output=Path(output)
    if output.exists() or output.is_symlink() or output.resolve()==source:
        raise FileExistsError('Encoder export requires a new output file')
    if any(parent.is_symlink() for parent in output.parents):
        raise ValueError('Encoder export cannot follow output symlinks')
    # Hash and deserialize the same open inode. Atomic replacement of latest.pt
    # by a live trainer cannot mix old tensors with the new file's digest.
    with source.open('rb') as stream:
        before=os.fstat(stream.fileno())
        digest=hashlib.sha256()
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
        source_digest=digest.hexdigest(); stream.seek(0)
        payload=torch.load(stream,map_location='cpu',weights_only=True)
        after=os.fstat(stream.fileno())
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('Checkpoint changed in place while reading; no export published')
    if (not isinstance(payload,dict) or payload.get('format_version')!=1
            or not isinstance(payload.get('metadata'),dict) or not _primitive(payload['metadata'])
            or not isinstance(payload.get('model_state_dict'),dict)
            or any(not isinstance(k,str) or not isinstance(v,torch.Tensor) for k,v in payload['model_state_dict'].items())):
        raise CheckpointError('Expected native tensor checkpoint with primitive metadata')
    prefix=component+'.'
    selected={key[len(prefix):]:value for key,value in payload['model_state_dict'].items() if key.startswith(prefix)}
    if not selected or any(not isinstance(value,torch.Tensor) for value in selected.values()):
        raise ValueError('Requested component is not a tensor encoder state in this native checkpoint')
    metadata=copy.deepcopy(payload['metadata'])
    metadata.update(source='native_encoder_export',export_component=component,
                    parent_checkpoint=str(source),parent_checkpoint_sha256=source_digest,
                    parent=copy.deepcopy(payload['metadata']))
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.encoder-',dir=output.parent);os.close(fd)
    try:
        with open(temporary,'wb') as stream:
            torch.save({'format_version':1,'model_state_dict':selected,'metadata':metadata},stream)
        # Atomic no-clobber publication, including a competing writer.
        export_digest=checkpoint_sha256(temporary)
        os.link(temporary,output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {'checkpoint':str(output.resolve()),'sha256':export_digest,'metadata':metadata}


def main(argv=None):
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--component',choices=('context_encoder','target_encoder','encoder'),required=True)
    args=parser.parse_args(argv)
    print(json.dumps(export_encoder(args.checkpoint,args.output,component=args.component),indent=2))


if __name__=='__main__': main()
