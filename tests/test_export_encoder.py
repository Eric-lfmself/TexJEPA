import pytest
import torch
from models import MiniViT, IJEPA, save_checkpoint
from models.backbones import build_backbone
from scripts.export_encoder import export_encoder


def test_native_component_exports_strictly_without_erasing_provenance(tmp_path):
    encoder=MiniViT(image_size=32,patch_size=8,embed_dim=16,depth=2,num_heads=2)
    model=IJEPA(encoder,predictor_dim=16,predictor_depth=1,num_heads=2)
    path=save_checkpoint(tmp_path/'full.pt',model,{'variant':'v3.1','epoch':201,'epoch_is_lineage_label':True,'mock':True,'evidence':'synthetic_smoke'})
    target=tmp_path/'encoder.pt'
    export_encoder(path,target,component='context_encoder')
    loaded=build_backbone({'backend':'native','checkpoint':{'path':str(target),'state_key':'model_state_dict'},
                          'image_size':32,'patch_size':8,'kwargs':{'embed_dim':16,'depth':2,'num_heads':2}})
    images=torch.rand(2,3,32,32)
    assert torch.equal(loaded.eval()(images),encoder.eval()(images))
    metadata=torch.load(target,weights_only=True)['metadata']
    assert metadata['mock'] is True and metadata['parent']['epoch_is_lineage_label'] is True
    with pytest.raises(FileExistsError):export_encoder(path,target,component='context_encoder')
    with pytest.raises(ValueError):export_encoder(path,tmp_path/'invalid.pt',component='encoder')
    assert not (tmp_path/'invalid.pt').exists()



def test_export_hash_and_weights_share_one_inode_when_latest_is_replaced(tmp_path,monkeypatch):
    from models.checkpoints import checkpoint_sha256
    path=tmp_path/'latest.pt';replacement=tmp_path/'new.pt'
    def payload(value):return {'format_version':1,'model_state_dict':{'context_encoder.weight':torch.full((2,2),value)},'metadata':{'source':'fixture'}}
    torch.save(payload(3.),path);torch.save(payload(7.),replacement)
    original_hash=checkpoint_sha256(path);loader=torch.load
    def replacing_loader(stream,*a,**k):
        result=loader(stream,*a,**k);replacement.replace(path);return result
    monkeypatch.setattr(torch,'load',replacing_loader)
    result=export_encoder(path,tmp_path/'encoder.pt',component='context_encoder')
    monkeypatch.setattr(torch,'load',loader)
    exported=torch.load(result['checkpoint'],weights_only=True)
    assert torch.all(exported['model_state_dict']['weight']==3.)
    assert result['metadata']['parent_checkpoint_sha256']==original_hash
    assert checkpoint_sha256(path)!=original_hash
