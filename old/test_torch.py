import torch
import torchvision
import sentence_transformers

print("Torch version:", torch.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
print("Torchvision version:", torchvision.__version__)
print("Sentence-Transformers version:", sentence_transformers.__version__)