import torch, clip
from torch import nn
from PIL import Image
from models.model import device


class CLIPBinary(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.clip = base_model                  # 이미 fp32
        self.classifier = nn.Linear(
            base_model.visual.output_dim, 2
        ).float()                               # ★ 헤드도 fp32 고정

    def forward(self, images):
        feats = self.clip.encode_image(images)  # (N,512) fp32
        return self.classifier(feats)


def get_image_model():
    print("Load Image Model")
    model, preprocess = clip.load("ViT-B/32", device = device)
    image_model = torch.load("ai_model/model_img.pt", map_location = device)

    model = CLIPBinary(model.float()).to(device)

    model.load_state_dict(image_model["state_dict"] if "state_dict" in image_model else image_model, strict=False)
    model.eval()
    print("Done")
    return model

model_img = get_image_model()