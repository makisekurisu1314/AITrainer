import torch
import torch.nn as nn
from models.hrnet_w18 import LightweightHRNet

device = "cpu"
model = LightweightHRNet(num_keypoints=21).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([100.0]).to(device)
)

# 固定一个假数据
dummy_img = torch.randn(1, 3, 224, 224).to(device)
dummy_hm = torch.zeros(1, 21, 64, 64).to(device)
dummy_hm[0, 0, 32, 32] = 1.0  # 第0个关键点在中心

for epoch in range(500):
    optimizer.zero_grad()
    out = model(dummy_img)
    loss = criterion(out, dummy_hm)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        pred_sigmoid = torch.sigmoid(out).detach()
        print(f"Epoch {epoch}: loss={loss.item():.6f}, "
              f"pred_at_32_32={pred_sigmoid[0,0,32,32].item():.4f}, "
              f"background_mean={pred_sigmoid[0,0,:,:].mean().item():.4f}")