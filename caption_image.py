import os
import sys
import random
from decoder_train import calc_clip_embedding
from PIL import Image
import torch
import open_clip
from dataset import IMAGE_FOLDER, TEXT_FOLDER, read_captions_json
from transformers import AutoTokenizer
from model_loader import CustomCLIPModel, load_custom_encoders, CLIPDecoder
from decoder_train import tokenize_and_encode, calc_clip_embedding
from config import CUSTOM_CLIP_FILE, WEIGHTS_PATH, device
import argparse
import chromadb
import numpy as np

CHROMA_DB_FILE = "embeddings_sismicos"
N_EMBEDS_BATCH = 4000

def save_embeddings(captions, encoded_captions):
    # Criar cliente local persistente
    client = chromadb.PersistentClient(path=CHROMA_DB_FILE)

    try:
        # Criar coleção (sem embedding_function)
        colecao = client.get_or_create_collection(name=CHROMA_DB_FILE)

        embeddings = encoded_captions.tolist()  # CLIP ViT-B/32 -> 512 dim
        if len(embeddings) == 0:
            print("Error. Empty embeddings.")
            sys.exit(1)

        # Process in batches
        for i in range(0, len(captions), N_EMBEDS_BATCH):
            batch_captions = captions[i:i + N_EMBEDS_BATCH]
            batch_embeddings = embeddings[i:i + N_EMBEDS_BATCH]
            batch_ids = [f"cap{j}" for j in range(i, i + len(batch_captions))]

            colecao.add(
                documents=batch_captions,
                embeddings=batch_embeddings,
                ids=batch_ids
            )

        print("Captions saved in ChromaDB.")
    except Exception as e:
        print(e)
        sys.exit(1)

def recover_embeddings():
    client = chromadb.PersistentClient(path=CHROMA_DB_FILE)

    # Carregar a coleção
    collection = client.get_collection(name=CHROMA_DB_FILE)

    # Pegar tudo de uma vez
    all_embeds = collection.get(
        include=[ "embeddings"]  # você escolhe o que trazer
    )
    return all_embeds["embeddings"].astype(np.float32)

def search_caption_embeds(captions, clip_encoder, tokenizer):
    try:
        print("Reading encoded captions...")
        numpy_embeddings = recover_embeddings()
        captions_embeddings = torch.from_numpy(numpy_embeddings)
    except:
        print("No caption embeddings found. Encoding captions...")
        captions_embeddings = encode_captions(captions, clip_encoder, tokenizer)
        save_embeddings(captions, captions_embeddings)
    return captions_embeddings.to(device)


@torch.no_grad
def encode_captions(captions, model, tokenizer):
    text_embeddings = torch.stack(
        [tokenize_and_encode(c, model, tokenizer) for c in captions]
    ).squeeze(1)
    return text_embeddings.to(device)

def generate_caption(image, tokenizer, clip_encoder, clip_decoder, transform):
    # start_token_id = tokenizer.convert_tokens_to_ids("this")
    start_token_id = tokenizer.cls_token_id
    end_token_id = tokenizer.sep_token_id

    print("Generating captions...")

    img_tensor = transform(image)

    with torch.no_grad():
        img_tensor = img_tensor.to(device)

        # Show image
        img_tensor = (
            img_tensor       - img_tensor.min()) / (
            img_tensor.max() - img_tensor.min())

        clip_embedding = calc_clip_embedding(clip_encoder, img_tensor.unsqueeze(0))

        predicted_tokens = clip_decoder.generate(
            clip_embedding, start_token_id, end_token_id, max_length=50)
        reconstructed = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
        return reconstructed

def load_encoder_and_decoder():
    torch.manual_seed(0)

    _, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )

    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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

    encoded_captions = search_caption_embeds(captions, clip_encoder, tokenizer)

    # vocab_size, encoded_captions, hidden_size=512, num_heads=8, temperature=0.7
    clip_decoder = CLIPDecoder(
        vocab_size, encoded_captions, temperature=clip_encoder.logit_scale
    ).to(device)
    clip_decoder.load_state_dict(torch.load(WEIGHTS_PATH))
    return tokenizer, clip_encoder, clip_decoder, preprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Captioner for seismic images.')

    parser.add_argument('-i', '--input_image', type=str, required=True,
                        help='Image used to generate caption.')

    args = parser.parse_args()

    tokenizer, clip_encoder, clip_decoder, preprocess = load_encoder_and_decoder()

    image = Image.open(args.input_image).convert("RGB")
    final_caption = generate_caption(
        image, tokenizer, clip_encoder, clip_decoder, preprocess)

    print("Final caption:", final_caption)
    image.show()
