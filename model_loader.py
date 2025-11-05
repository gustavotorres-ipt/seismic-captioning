import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchvision import models
from config import VISION_MODEL, LANGUAGE_MODEL

class CustomCLIPModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, init_temperature=0.1):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor([torch.log(torch.tensor(1.0 / init_temperature))]))
        self.image_encoder = image_encoder  # Your custom image encoder
        self.text_encoder = text_encoder  # Your custom text encoder
        self.projection_layer = nn.Linear(in_features=768, out_features=512)

    def encode_image(self, images):
        features_image = self.image_encoder(images)[:,:,0,0]
        return features_image
        # return self.projection_layer(image_features)  # Project to CLIP space

    def encode_text(self, tokenized_texts):
        output_llm = self.text_encoder(**tokenized_texts)
        features_text = output_llm.last_hidden_state[:, 0, :]
        features_proj = self.projection_layer(features_text)   # Custom text encoder
        return features_proj
        # return self.projection_layer(text_features)  # Project to CLIP space

    def forward(self, image, tokenized_text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(tokenized_text)
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Compute cosine similarity
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T

        # Use learned temperature
        logit_scale = self.logit_scale.exp()
        logits_per_image *= logit_scale
        logits_per_text *= logit_scale

        return logits_per_image, logits_per_text


class CLIPDecoder(nn.Module):
    def __init__(self, vocab_size, encoded_captions, hidden_size=512, num_heads=4, temperature=0.7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.out = nn.Linear(hidden_size, vocab_size)
        self.text_memory = encoded_captions  # This is your text embedding memory
        self.temperature = temperature

    @torch.no_grad()
    def clip_proj(self, img_embed):
        img_embed = img_embed.unsqueeze(0)
        img_embed = F.normalize(img_embed, dim=-1)
        text_embeddings = F.normalize(self.text_memory, dim=-1)

        sim = torch.matmul(img_embed, text_embeddings.T)
        weights = F.softmax(sim / self.temperature, dim=-1)
        v_proj = torch.matmul(weights, text_embeddings)
        v_proj_norm = F.normalize(v_proj, dim=-1)  # [1, D]
        return v_proj_norm  # shape: [1, D]

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward_from_embedding(self, img_clip_embeds, tgt_tokens, debug=True):
        batch_size = img_clip_embeds.size(0)
        device = img_clip_embeds.device

        prefix_emb = self.clip_proj(img_clip_embeds)  # [1, D]
        memory = prefix_emb.squeeze(0)#.expand(batch_size, 1, -1)  # [B, 1, D]

        tgt_emb = self.embedding(tgt_tokens)  # [B, T, H]

        if debug:
            print("\nEmbedding do target (decoder input):\n", tgt_emb)

        tgt_mask = self.generate_square_subsequent_mask(tgt_tokens.size(1)).to(device)

        # Decoder expects: tgt=[B, T, H], memory=[B, M, H]
        dec_out = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)

        logits = self.out(dec_out)  # [B, T, vocab_size]
        if debug:
            print("\nLogits finais sobre o vocabulário:\n", logits)

        predicted_tokens = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)
        return logits, predicted_tokens

    def generate(self, img_clip_embeds, start_token_id, end_token_id, max_length=50):
        device = img_clip_embeds.device
        batch_size = img_clip_embeds.size(0)

        # Start with <BOS>
        generated = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)

        prefix_emb = self.clip_proj(img_clip_embeds)
        memory = prefix_emb.squeeze(0)#.unsqueeze(1).expand(batch_size, 1, -1)  # [B, 1, H]

        for _ in range(max_length):
            tgt_emb = self.embedding(generated)  # [B, T, H]
            tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(device)

            dec_out = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            logits = self.out(dec_out)  # [B, T, vocab_size]
            next_token_logits = logits[:, -1, :]  # [B, vocab_size]

            # Sampling instead of argmax might help too
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)

            if torch.all(next_token.squeeze() == end_token_id):
                break

        return generated

def load_custom_encoders():
    # image_encoder
    # resnet18 = models.resnet18(pretrained=False)
    # image_encoder = nn.Sequential(
    #     resnet18.conv1,
    #     resnet18.bn1,
    #     resnet18.relu,
    #     resnet18.maxpool,
    #     resnet18.layer1,
    #     resnet18.layer2,
    #     resnet18.layer3,
    #     resnet18.layer4,
    #     resnet18.avgpool
    # )
    if "resnet50" in VISION_MODEL:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    text_encoder = AutoModel.from_pretrained(LANGUAGE_MODEL)

    image_encoder = nn.Sequential(*list(model.children())[:-1])
    image_encoder.load_state_dict(torch.load(VISION_MODEL))  # Custom image encoder
    image_encoder.eval()

    return image_encoder, text_encoder

def load_clip_model():
    image_encoder, text_encoder = load_custom_encoders()
    custom_clip_model = CustomCLIPModel(image_encoder, text_encoder)

    return custom_clip_model

