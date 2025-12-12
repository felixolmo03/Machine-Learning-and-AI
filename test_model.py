import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from transformers import GPT2Tokenizer

# ============================================================================
# MODEL DEFINITION (needed to load the weights)
# ============================================================================

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_length: int = 256
    hidden_size: int = 384
    num_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    dropout: float = 0.1


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.head_dim)
        
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_length, config.max_seq_length), diagonal=1).bool()
        )
    
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.mlp(self.ln2(x))
        return x


class StoryModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
    
    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        
        tok_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(T, device=input_ids.device))
        x = self.dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), 
                                   labels[:, 1:].reshape(-1))
        
        return {"logits": logits, "loss": loss}
    
    @torch.no_grad()
    def generate(self, input_ids, max_length=200, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_length - input_ids.size(1)):
            input_ids_cond = input_ids[:, -self.config.max_seq_length:]
            logits = self.forward(input_ids_cond)["logits"][:, -1, :]
            logits = logits / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# ============================================================================
# LOAD AND RUN
# ============================================================================

def main():
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load("story_model.pt", map_location=device, weights_only=False)
    model = StoryModel(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    print("Model loaded!\n")
    
    # Generation function
    def generate_story(prompt, max_length=300, temperature=0.8):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(input_ids, max_length=max_length, temperature=temperature)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Interactive loop
    print("="*50)
    print("STORY GENERATOR")
    print("="*50)
    print("Enter a prompt to generate a story.")
    print("Type 'quit' to exit.\n")
    
    while True:
        prompt = input("Prompt: ")
        if prompt.lower() == 'quit':
            print("Goodbye!")
            break
        
        print("\nGenerating...\n")
        print("-"*50)
        print(generate_story(prompt))
        print("-"*50 + "\n")

if __name__ == "__main__":
    main()