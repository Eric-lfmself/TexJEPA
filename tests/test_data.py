"""Data integrity checks: seed stability, bbox alignment and leakage-free split."""
import json
import torch
import pytest
from PIL import Image
from data import DummyCXRDataset, LocalManifestCXRDataset, collate_cxr, deterministic_split


def test_dummy_contract_reproducibility():
    ds = DummyCXRDataset(size=6)
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    a = ds[2]
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(a["image"], ds[2]["image"])
    assert a["labels"].shape == (15,) and len(a["boxes"]) == 2
    for box in a["boxes"]:
        assert a["labels"][box["class_index"]] == 1
    assert collate_cxr([ds[0], ds[1]])["image"].shape == (2, 3, 32, 32)


def test_split_stable_by_image_id():
    ids=[f"id-{i}" for i in range(15000)]
    a,b=deterministic_split(ids)
    assert len(a)==12000 and len(b)==3000 and not set(a)&set(b)
    rev=list(reversed(ids)); ar,br=deterministic_split(rev)
    assert {ids[i] for i in a} == {rev[i] for i in ar}
    with pytest.raises(ValueError, match="Duplicate"):
        deterministic_split(["a", "a"],1,1)


def test_local_resize_and_mapping(tmp_path):
    Image.new("RGB", (40,20)).save(tmp_path/"example.png")
    record={"image_id":"a","image":"example.png","labels":[1]+[0]*14,
            "boxes":[{"class_index":0,"bbox":[10,5,30,15]}]}
    content={"class_names":[f"c{i}" for i in range(15)],"samples":[record]}
    manifest=tmp_path/"manifest.json"; manifest.write_text(json.dumps(content))
    ds=LocalManifestCXRDataset(manifest,tmp_path,image_size=32)
    assert ds[0]["boxes"][0]["bbox"] == [8,8,24,24]
    record["image"]="../outside.png"; manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="confined"):
        LocalManifestCXRDataset(manifest,tmp_path)
