# service/app.py
import os
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
import yaml

from ner.infer import NERPipeline
from ner.postprocess import boost_numeric_entities
from ner.postprocess import postprocess_all

app = FastAPI(title="X5 NER Service", version="0.2.0")


class PredictIn(BaseModel):
    input: str


# ленивое создание пайплайна
_PIPE = {"obj": None}


def get_pipe():
    if _PIPE["obj"] is None:
        cfg_path = os.environ.get("SERVICE_CONFIG", "configs/service.yaml")
        if os.path.exists(cfg_path):
            cfg = yaml.safe_load(open(cfg_path))
            model_path = cfg.get("model_path", "artifacts/ner-checkpoint")
            max_len = int(cfg.get("max_seq_len", 128))
        else:
            model_path = "artifacts/ner-checkpoint"
            max_len = 128
        _PIPE["obj"] = NERPipeline(model_path, max_len=max_len)
    return _PIPE["obj"]


@app.post("/api/predict")
def predict(inp: PredictIn) -> List[Dict]:
    text = (inp.input or "").strip()
    if not text:
        return []
    pipe = get_pipe()
    ents = pipe.predict_entities(text)  # [(s,e,'B-XXX'), ...]
    ents = postprocess_all(text, ents, do_split_type=True, do_boost_numeric=True)
    return [{"start_index": s, "end_index": e, "entity": tag} for (s, e, tag) in ents]
