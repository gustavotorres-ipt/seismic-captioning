import torch

N_EPOCHS = 40
CUSTOM_CLIP_FILE = 'clip_seismic_facies.pth'
WEIGHTS_PATH = 'decoder_seismic_facies.pth'

CHROMA_DB_FILE = 'embeddings_sismicos'
N_EMBEDS_BATCH = 1000

VISION_MODEL = 'resnet34_sismofacies.pth'
LANGUAGE_MODEL = 'mlm_sismofacies.pt'

IMAGE_FOLDER = 'imagens_janelas_sismofacies/training'
TEXT_FOLDER = 'legendas_sismofacies/training'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
