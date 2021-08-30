import torch
from torch_em.trainer import DefaultTrainer

trainer = DefaultTrainer.from_checkpoint("./checkpoints/affinity_model", device="cpu")
model = trainer.model
torch.save(model, "./checkpoints/affinity_model/model.pt")
