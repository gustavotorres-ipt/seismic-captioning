from PIL import Image
from caption_image import generate_caption

class CaptionGenerator:
    def __init__(self, tokenizer, clip_encoder, clip_decoder, preprocess):
        self.tokenizer = tokenizer
        self.clip_encoder = clip_encoder
        self.clip_decoder = clip_decoder
        self.preprocess = preprocess

    def gerar_legenda(self, caminho_imagem: str) -> str:
        # Exemplo simples (substitua pelo seu código de IA)

        image = Image.open(caminho_imagem).convert("RGB")
        final_caption = generate_caption(
            image, self.tokenizer, self.clip_encoder,
            self.clip_decoder, self.preprocess)
        return final_caption
