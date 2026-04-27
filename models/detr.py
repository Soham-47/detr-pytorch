import torch
from torch import nn
from .backbone.backbone import Backbone
from .transformer.transformer import Transformer
from .heads.prediction import PredictionHead

class DETR(nn.Module):
    def __init__(self, num_classes=91, embed_dim=256, num_heads=8, dim_feedforward=2048, num_layers=6, num_queries=100):
        super().__init__()
        
        self.backbone = Backbone(embed_dim)
        
        self.transformer = Transformer(embed_dim, num_heads, dim_feedforward, num_layers)
        
        self.prediction_head = PredictionHead(embed_dim, num_classes, num_heads)
        
        self.query_embed = nn.Embedding(num_queries, embed_dim)

    def forward(self, x):
        
        features, pos = self.backbone(x)
        
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(x.size(0), 1, 1)
        
        hs = self.transformer(features, query_pos, pos)
        
        logits, boxes = self.prediction_head(hs)
        
        return {"pred_logits": logits, "pred_boxes": boxes}

if __name__ == "__main__":
    model = DETR(num_classes=20, embed_dim=256, num_heads=8, dim_feedforward=2048, num_layers=6)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("DETR wired and initialized successfully!")
    print(f"Pred Logits Shape: {output['pred_logits'].shape}")
    print(f"Pred Boxes Shape: {output['pred_boxes'].shape}")
