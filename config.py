import torch

N_EPOCHS = 40
CUSTOM_CLIP_FILE = "customized_clip.pth"
WEIGHTS_PATH = "decoder_model.pth"

CHROMA_DB_FILE = "embeddings_sismicos"
N_EMBEDS_BATCH = 1000



device = "cuda" if torch.cuda.is_available() else "cpu"
