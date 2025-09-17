import os
import sys
import numpy as np
import chromadb
from decoder_train import calc_clip_embedding
import torch
from decoder_train import tokenize_and_encode, calc_clip_embedding
from config import CUSTOM_CLIP_FILE, WEIGHTS_PATH, device, CHROMA_DB_FILE, N_EMBEDS_BATCH
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap, QFont
from generator_caption import CaptionGenerator
from caption_image import load_encoder_and_decoder


# def load_encoder_and_decoder():
#     torch.manual_seed(0)
# 
#     _, _, preprocess = open_clip.create_model_and_transforms(
#         'ViT-B-32', pretrained='laion2b_s34b_b79k'
#     )
# 
#     print("Loading models...")
#     tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# 
#     image_encoder, text_encoder = load_custom_encoders()
#     clip_encoder = CustomCLIPModel(image_encoder, text_encoder)
#     clip_encoder.load_state_dict(torch.load(CUSTOM_CLIP_FILE))
# 
#     clip_encoder.to(device)
#     vocab_size = tokenizer.vocab_size
# 
#     # load captions
#     caption_files = [
#         os.path.join(TEXT_FOLDER, caption_file)
#         for caption_file in os.listdir(TEXT_FOLDER)
#     ]
#     random.shuffle(caption_files)
#     captions = [read_captions_json(caption_file)
#                 for caption_file in caption_files]
# 
#     encoded_captions = search_caption_embeds(captions, clip_encoder, tokenizer)
# 
#     # vocab_size, encoded_captions, hidden_size=512, num_heads=8, temperature=0.7
#     clip_decoder = CLIPDecoder(
#         vocab_size, encoded_captions, temperature=clip_encoder.logit_scale
#     ).to(device)
#     clip_decoder.load_state_dict(torch.load(WEIGHTS_PATH))
#     return tokenizer, clip_encoder, clip_decoder, preprocess


class ImageCaptionApp(QWidget):
    def __init__(self, caption_generator):
        super().__init__()

        self.caption_generator = caption_generator

        self.setWindowTitle("Exibição de Imagem Sísmica com Legenda")
        self.setGeometry(200, 200, 600, 400)
        font = QFont("Arial", 14)

        # Widgets
        self.image_label = QLabel("Nenhuma imagem carregada")
        self.image_label.setScaledContents(True)
        self.image_label.setFont(font)

        self.caption_label = QLabel("Legenda aparecerá aqui")
        self.caption_label.setWordWrap(True)

        self.caption_label.setFont(font)

        self.load_button = QPushButton("Carregar Imagem")
        self.load_button.setFont(font)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.caption_label)
        layout.addWidget(self.load_button)

        self.setLayout(layout)

        # Conexão
        self.load_button.clicked.connect(self.load_image)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Abrir Imagem", "", "Imagens (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(400, 300))

            legenda = self.travar_form_e_gerar_legenda(file_name)

            self.caption_label.setText(legenda)
            self.load_button.setDisabled(False)
            self.repaint()

    def travar_form_e_gerar_legenda(self, file_name):
        # Disable load button and change text to loading image...
        self.load_button.setDisabled(True)
        self.caption_label.setText("Gerando legenda...")
        self.repaint()

        # Gera a legenda automaticamente
        legenda = self.caption_generator.gerar_legenda(file_name)
        return legenda


if __name__ == "__main__":
    tokenizer, clip_encoder, clip_decoder, preprocess = load_encoder_and_decoder()

    app = QApplication(sys.argv)
    caption_generator = CaptionGenerator(
        tokenizer, clip_encoder, clip_decoder, preprocess)
    window = ImageCaptionApp(caption_generator)
    window.show()
    sys.exit(app.exec_())
