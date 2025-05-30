import os
import timm
import torch
import torch.nn as nn
import torchvision
from transformers import BertModel, GPT2Model, DistilBertModel


def get_pretrained_model(arch, data_name, data_dir):
    if arch == "resnet":
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    elif arch == "clip":
        model = timm.create_model(
            "vit_base_patch32_clip_224.openai", pretrained=True, num_classes=0
        )
    elif arch == "bert":
        model = BertModel.from_pretrained("bert-base-uncased")
        model = BertFeatureWrapper(model)
    elif arch == "bert-ft":
        if data_name == "multinli":
            model = DistilBertModel.from_pretrained(
                "ishan/distilbert-base-uncased-mnli"
            )
            model = BertFeatureWrapper(model)
        elif data_name == "civilcomments":
            model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            model_path = os.path.join(data_dir, "civilcomments", "best_model.pth")
            state_dict = torch.load(model_path)
            for k in list(state_dict["algorithm"].keys()):
                new_k = k.replace("model.distilbert.", "")
                state_dict["algorithm"][new_k] = state_dict["algorithm"].pop(k)
                if "classifier" in new_k:
                    state_dict["algorithm"].pop(new_k)
            model.load_state_dict(state_dict["algorithm"])
            model = BertFeatureWrapper(model)
    elif arch == "gpt2":
        model = GPT2Model.from_pretrained("gpt2")
        model = BertFeatureWrapper(model)
    else:
        raise ValueError(f"Architecture {arch} not supported")

    return model


class BertFeatureWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_outputs = model.config.hidden_size
        # classifier_dropout = (
        #     hparams["last_layer_dropout"]
        #     if hparams["last_layer_dropout"] != 0.0
        #     else model.config.hidden_dropout_prob
        # )
        # self.dropout = nn.Dropout(classifier_dropout)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        kwargs = {"input_ids": x[:, :, 0], "attention_mask": x[:, :, 1]}
        if x.shape[-1] == 3:
            kwargs["token_type_ids"] = x[:, :, 2]
        output = self.model(**kwargs)
        if hasattr(output, "pooler_output"):
            return self.dropout(output.pooler_output)
        else:
            return self.dropout(output.last_hidden_state[:, 0, :])
