import torch
import json
import sys
import os
import torch.nn as nn

EMBEDDING_DIM = 768 # Размерность эмбеддингов paraphrase-multilingual-mpnet-base-v2
PROJECTOR_OUT_DIM = 2560 # Размерность эмбеддингов Gemma (для gemma-3-4b-pt)
DEVICE = "mps"

class Projector(nn.Module):
    def __init__(self, in_dim=EMBEDDING_DIM, out_dim=PROJECTOR_OUT_DIM):
        super().__init__()
        # Проектор из 768 (ST) в 2560 (Gemma)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.GELU(), # Используем GELU, как часто делают в трансформерах
            nn.Linear(1024, out_dim)
        )

    def forward(self, x, **kwargs):
        return self.mlp(x)


def process_input(projector):
    print("READY", flush=True)

    while True:
        readable, _, _ = select.select([sys.stdin], [], [], 1.0)
        if readable:
            line = sys.stdin.readline().strip()
            if not line:
                print("EOF received, exiting", file=sys.stderr, flush=True)
                break
            try:
                data = json.loads(line)
                if "context_embedding" in data and isinstance(data["context_embedding"], list):
                    embedding = data["context_embedding"]
                    projected_embedding = projector(embedding).to(DEVICE)
                    if projected_embedding is not None:
                        result = {"type": "projected_embedding", "embedding": projected_embedding}
                        print(json.dumps(result), flush=True)
                        print(f"Generated embeddings shape: {projected_embedding.shape}", file=sys.stderr, flush=True)
                        print("END", flush=True)
                    else:
                        print("Failed to project embedding", file=sys.stderr, flush=True)
                else:
                    print("Invalid input format", file=sys.stderr, flush=True)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    projector = Projector(in_dim=EMBEDDING_DIM, out_dim=PROJECTOR_OUT_DIM).to(DEVICE)
    base_model_path = os.environ.get('MODEL_DIR') or os.path.dirname(__file__)
    encoder_path = os.path.join(base_model_path, "checkpoints", "projector_epoch0_step04200.pt")  # Adjust path to your model directory
    projector.load_state_dict(torch.load("projector_epoch0_step04200.pt"))

    process_input(projector)
