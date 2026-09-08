import json
from pathlib import Path
import pytest

def test_merge_rejects_duplicate(tmp_path, monkeypatch):
    from scripts.merge_adaptive_vsv_shards import main
    ids=tmp_path/'ids'; ids.write_text('1\n'); shard=tmp_path/'a'; row={'image_id':1,'base_lambda':0.0}; shard.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
    monkeypatch.setattr('sys.argv',['merge','--shards',str(shard),'--image-id-file',str(ids),'--lambda-grid','0.0','--output',str(tmp_path/'out')])
    with pytest.raises(ValueError): main()
