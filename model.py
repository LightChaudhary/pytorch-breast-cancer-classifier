import torch
import torch.nn as nn

class CancerClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(30,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, x):
        return self.network(x)

# if __name__ == "__main__":
#     model = CancerClassifier()

#     x = torch.randn(32,30)
#     output = model(x)

#     print("Input shape:", x.shape)
#     print("Output shape:", output.shape)
#     print("Sample logits:", output[:5])

#     total_params = sum(
#         p.numel() for p in model.parameters() if p.requires_grad
#     )

#     print("Trainable parameters:", total_params)