import os
import random
from decoder_train import calc_clip_embedding
from PIL import Image
import torch
import open_clip
from dataset import IMAGE_FOLDER, TEXT_FOLDER, read_captions_json
from transformers import AutoTokenizer
from model_loader import CustomCLIPModel, load_custom_encoders, CLIPDecoder
from decoder_train import tokenize_and_encode, calc_clip_embedding
import argparse


CUSTOM_CLIP_FILE = "customized_clip.pth"
WEIGHTS_PATH = "decoder_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad
def encode_captions(captions, model, tokenizer):
    text_embeddings = torch.stack(
        [tokenize_and_encode(c, model, tokenizer) for c in captions]
    ).squeeze(1)
    return text_embeddings.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Criador de volume de labels')

    parser.add_argument('-i', '--input_image', type=str, required=True,
                        help='Imagem utilizada para gerar legenda')

    args = parser.parse_args()

    torch.manual_seed(0)
    _, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )

    print("Tokenizing words...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("Loading model...")
    image_encoder, text_encoder = load_custom_encoders()
    clip_encoder = CustomCLIPModel(image_encoder, text_encoder)
    clip_encoder.load_state_dict(torch.load(CUSTOM_CLIP_FILE))

    clip_encoder.to(device)
    vocab_size = tokenizer.vocab_size

    # load captions
    caption_files = [
        os.path.join(TEXT_FOLDER, caption_file)
        for caption_file in os.listdir(TEXT_FOLDER)
    ]
    random.shuffle(caption_files)
    captions = [read_captions_json(caption_file)
                for caption_file in caption_files]

    print("Encoding captions...")
    encoded_captions = encode_captions(captions, clip_encoder, tokenizer)

    # vocab_size, encoded_captions, hidden_size=512, num_heads=8, temperature=0.7
    clip_decoder = CLIPDecoder(
        vocab_size, encoded_captions, temperature=clip_encoder.logit_scale
    ).to(device)
    clip_decoder.load_state_dict(torch.load(WEIGHTS_PATH))

    start_token_id = tokenizer.convert_tokens_to_ids("this")
    # start_token_id = tokenizer.cls_token_id
    end_token_id = tokenizer.sep_token_id

    print("\n=== Geração de legenda ===")

    image = Image.open(args.input_image).convert("RGB")
    img_tensor = preprocess(image)

    with torch.no_grad():
        img_tensor = img_tensor.to(device)

        # Show image
        img_tensor = (
            img_tensor       - img_tensor.min()) / (
            img_tensor.max() - img_tensor.min())
        image.show()

        clip_embedding = calc_clip_embedding(clip_encoder, img_tensor.unsqueeze(0))

        predicted_tokens = clip_decoder.generate(
            clip_embedding, start_token_id, end_token_id, max_length=50)
        reconstructed = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
        print("Texto gerado:", reconstructed)
