import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import IMAGE_FOLDER, TEXT_FOLDER, load_datasets, read_captions_json
from transformers import AutoTokenizer, AutoModel
from model_loader import CustomCLIPModel, load_custom_encoders, CLIPDecoder
from torchvision.transforms.functional import to_pil_image


N_EPOCHS = 30
CUSTOM_CLIP_FILE = "customized_clip.pth"
WEIGHTS_PATH = "decoder_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad
def tokenize_and_encode(caption, model, tokenizer):
    tokenized_captions = tokenizer(
        caption, padding=True, truncation=True, return_tensors="pt",
        max_length=128
    ).to(device)
    return model.encode_text(tokenized_captions)

@torch.no_grad
def encode_captions(dataset, model, tokenizer):
    text_embeddings = torch.stack(
        [tokenize_and_encode(c[1], model, tokenizer) for c in dataset]
    ).squeeze(1)
    return text_embeddings.to(device)


@torch.no_grad
def calc_clip_embedding(clip_encoder, img_input):

    # clip_embedding = clip_encoder.encode_text(text_input)  # [1, 512]
    clip_embedding = clip_encoder.encode_image(img_input)  # [1, 512]
    clip_embedding = clip_embedding.unsqueeze(1)  # [1, seq_len=1, hidden_size]

    return clip_embedding


def run_train_epoch(clip_encoder, clip_decoder, tokenizer, train_loader):
    total_loss = 0

    for image_batch, text_batch in tqdm(train_loader):
        image_batch = image_batch.to(device)

        text_inputs = tokenizer(
            text_batch, padding=True, truncation=True,
            return_tensors="pt", max_length=128
        ).to(device)
        target_tokens = text_inputs.input_ids

        decoder_input_ids = target_tokens[:, :-1]
        target_ids = target_tokens[:, 1:]

        clip_embedding = calc_clip_embedding(clip_encoder, image_batch)

        logits, predicted_tokens = clip_decoder.forward_from_embedding(
            clip_embedding, decoder_input_ids, debug=False)

        # print("Real:", tokenizer.convert_ids_to_tokens(target_ids[0]))
        # print("Predicted:", tokenizer.convert_ids_to_tokens(predicted_tokens[0]))
        # print("-----------------------------------------------")

        loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(train_loader)

@torch.no_grad
def run_val_epoch(clip_encoder, clip_decoder, tokenizer, val_loader):
    total_loss = 0

    for image_batch, text_batch in tqdm(val_loader):
        image_batch = image_batch.to(device)

        text_inputs = tokenizer(
            text_batch, padding=True, truncation=True,
            return_tensors="pt", max_length=128
        ).to(device)

        target_tokens = text_inputs.input_ids
        decoder_input_ids = target_tokens[:, :-1]
        target_ids = target_tokens[:, 1:]

        clip_embedding = calc_clip_embedding(clip_encoder, image_batch)

        logits, predicted_tokens = clip_decoder.forward_from_embedding(
            clip_embedding, decoder_input_ids, debug=False)

        # print("Real:", tokenizer.convert_ids_to_tokens(target_ids[0]))
        #print("Predicted:", tokenizer.convert_ids_to_tokens(predicted_tokens[0]))
        #print("-----------------------------------------------")

        loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
        total_loss += loss.item()

    return total_loss / len(val_loader)


if __name__ == "__main__":


    # Configurações
    input_do_usuario = "A model with a fault to the east"

    clip_encoder, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )

    train_dataset, val_dataset, test_dataset = load_datasets(preprocess)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Tokenizing words...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("Loading model...")
    image_encoder, text_encoder = load_custom_encoders()
    clip_encoder = CustomCLIPModel(image_encoder, text_encoder)
    clip_encoder.load_state_dict(torch.load(CUSTOM_CLIP_FILE))

    clip_encoder.to(device)
    vocab_size = tokenizer.vocab_size

    print("Encoding captions...")
    encoded_captions = encode_captions(train_dataset, clip_encoder, tokenizer)

    # vocab_size, encoded_captions, hidden_size=512, num_heads=8, temperature=0.7
    clip_decoder = CLIPDecoder(
        vocab_size, encoded_captions, temperature=clip_encoder.logit_scale
    ).to(device)

    # Check if weights exist and perform training if they don't

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(clip_decoder.parameters(), lr=1e-4)

    if WEIGHTS_PATH in os.listdir("."):
        clip_decoder.load_state_dict(torch.load(WEIGHTS_PATH))

    else:

        print("Training...")
        for epoch in range(N_EPOCHS):

            # Treino
            avg_loss_train = run_train_epoch(
                clip_encoder, clip_decoder, tokenizer, train_loader)
            # Validação
            avg_loss_val = run_val_epoch(
                clip_encoder, clip_decoder, tokenizer, val_loader)

            print(f"Epoch {epoch+1}, Training Loss: {avg_loss_train:.4f},",
                  f"Val loss: {avg_loss_val:.4f}")

        # Save the model's state_dict
        torch.save(clip_decoder.state_dict(), WEIGHTS_PATH)
        print(WEIGHTS_PATH, "saved.")

    start_token_id = tokenizer.convert_tokens_to_ids("this")
    #start_token_id = tokenizer.cls_token_id
    end_token_id = tokenizer.sep_token_id

    print("\n=== Geração de legenda ===")
    with torch.no_grad():
        test_sample = next(iter(test_loader))

        idx_sample = random.randint(0, 31)
        img_tensor = test_sample[0][idx_sample].to(device)

        test_caption = test_sample[1][idx_sample]

        # Show image
        image = (
            img_tensor       - img_tensor.min()) / (
            img_tensor.max() - img_tensor.min()
        )
        image_pil = to_pil_image(image)
        image_pil.show()

        clip_embedding = calc_clip_embedding(clip_encoder, img_tensor.unsqueeze(0))

        print("\nTexto original:", test_caption)

        predicted_tokens = clip_decoder.generate(
            clip_embedding, start_token_id, end_token_id, max_length=50)
        reconstructed = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
        print("Texto gerado:", reconstructed)
