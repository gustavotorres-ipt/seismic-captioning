import torch

N_EPOCHS = 40
CUSTOM_CLIP_FILE = "customized_clip.pth"
WEIGHTS_PATH = "decoder_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
