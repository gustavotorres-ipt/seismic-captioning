import os
import random
import torch
import open_clip
from transformers import AutoTokenizer, AutoModel
from model_loader import CustomCLIPModel, load_custom_encoders
from dataset import IMAGE_FOLDER, TEXT_FOLDER, load_datasets, read_captions_json
from PIL import Image
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch import nn

N_CAPTIONS_REF = 500

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MappingMLP(nn.Module):
    def __init__(self, input_dim=512, output_dim=768, prefix_len=10):
        super().__init__()
        self.prefix_len = prefix_len
        self.output_dim = output_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim * prefix_len),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.proj(x)  # (output_dim * prefix_len)
        return out.view(self.prefix_len, self.output_dim)


def tokenize_and_encode(caption, model, tokenizer):
    tokenized_captions = tokenizer(
        caption, padding=True, truncation=True, return_tensors="pt",
        max_length=128
    ).to(DEVICE)
    return model.encode_text(tokenized_captions)

def calc_projection_vector(captions, image, model, tokenizer):
    with torch.no_grad():
        img_embed = model.encode_image(image.unsqueeze(0))

        temperature = model.logit_scale

        text_embeddings = torch.stack(
            [tokenize_and_encode(c, model, tokenizer) for c in captions]
        ).squeeze(1)
        # Ensure both are normalized
        img_embed = F.normalize(img_embed, dim=-1)      # [B, D]
        text_embeddings = F.normalize(text_embeddings, dim=-1)  # [N, D]

        # Compute similarities: [B, N]
        sim = torch.matmul(img_embed, text_embeddings.T)  # cosine similarity

        # Softmax over memory items (along N)
        weights = F.softmax(sim / temperature, dim=-1)      # [B, N]

        # Weighted sum of memory items: [B, D]
        v_proj = torch.matmul(weights, text_embeddings)

        # L2-normalize the final projection
        v_proj_norm = F.normalize(v_proj, dim=-1)

        return v_proj_norm


def load_captions():
    captions_files = [os.path.join(TEXT_FOLDER, f) for f in os.listdir(TEXT_FOLDER)]
    random.shuffle(captions_files)

    captions = [read_captions_json(path) for path in captions_files[:N_CAPTIONS_REF]]
    return captions


def decode_embeddings(prompt, v_proj):
    ## Load GPT model & tokenizer (distilgpt2)
    gpt_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(DEVICE)
    gpt_model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

    # Set pad token (since distilgpt2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token

    prefix_len = 10
    mapper = MappingMLP(prefix_len=prefix_len).to(DEVICE)
    with torch.no_grad():
        prefix_embeds = mapper(v_proj)  # (prefix_len, 768)

        tokens = tokenizer(prompt, return_tensors='pt').to(DEVICE)
        token_embeds = gpt_model.transformer.wte(tokens.input_ids).squeeze(0)  # shape: (L, 512)

        # ----- Combine prefix and token embeddings -----
        full_embed = torch.cat([prefix_embeds, token_embeds], dim=0).unsqueeze(0)  # shape: (1, L+prefix_len, 512)

        # ----- Generate Caption -----
        output_ids = gpt_model.generate(
            inputs_embeds=full_embed,
            max_length=50,
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode and show result
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nGenerated Caption:")
    print(caption)


if __name__ == "__main__":
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )

    train_dataset, val_dataset, test_dataset = load_datasets(preprocess)

    print("Tokenizing words...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("Loading model...")
    image_encoder, text_encoder = load_custom_encoders()
    model = CustomCLIPModel(image_encoder, text_encoder)
    model.load_state_dict(torch.load("customized_clip.pth"))

    model.to(DEVICE)

    captions = load_captions()

    image_tensor = test_dataset[0][0].to(DEVICE)
    # caption = test_dataset[0][1]
    v_proj = calc_projection_vector(captions, image_tensor, model, tokenizer)
    breakpoint()

    # decode_embeddings(captions[50], v_proj)
