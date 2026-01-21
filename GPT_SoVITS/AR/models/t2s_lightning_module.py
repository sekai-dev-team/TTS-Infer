# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_lightning_module.py
# reference: https://github.com/lifeiteng/vall-e
import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
from typing import Dict

import torch
from torch.nn import Module

from AR.models.t2s_model import Text2SemanticDecoder

class Text2SemanticLightningModule(Module):
    def __init__(self, config, output_dir, is_train=True):
        super().__init__()
        self.config = config
        self.top_k = 3
        self.model = Text2SemanticDecoder(config=config, top_k=self.top_k)
        pretrained_s1 = config.get("pretrained_s1")
        if pretrained_s1 and is_train:
            # print(self.load_state_dict(torch.load(pretrained_s1,map_location="cpu")["state_dict"]))
            print(
                self.load_state_dict(
                    torch.load(
                        pretrained_s1,
                        map_location="cpu",
                        weights_only=False,
                    )["weight"],
                )
            )
        if is_train:
            # self.automatic_optimization = False
            # self.save_hyperparameters()
            self.eval_dir = output_dir / "eval"
            self.eval_dir.mkdir(parents=True, exist_ok=True)

    def forward(self, batch: Dict):
        # basic forward for inference or simple loss calculation if needed (but we removed training logic)
        forward = self.model.forward if self.config["train"].get("if_dpo", False) == True else self.model.forward_old
        return forward(
            batch["phoneme_ids"],
            batch["phoneme_ids_len"],
            batch["semantic_ids"],
            batch["semantic_ids_len"],
            batch["bert_feature"],
        )