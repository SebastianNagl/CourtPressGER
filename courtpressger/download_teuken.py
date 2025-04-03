import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

# Lade Umgebungsvariablen
load_dotenv()

# Hugging Face API Token
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY nicht in .env gefunden!")

# Modell-ID
MODEL_ID = "openGPT-X/Teuken-7B-instruct-research-v0.4"

# Speicherort für das Modell
MODEL_PATH = "models/teuken"

# Erstelle das Verzeichnis, falls es nicht existiert
os.makedirs(MODEL_PATH, exist_ok=True)

print("Lade Teuken-7B-Instruct Modell...")

# Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_API_KEY,
    trust_remote_code=True
)

# Modell laden
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_API_KEY,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("Modell erfolgreich geladen!")

# Speichere Modell und Tokenizer
print(f"Speichere Modell und Tokenizer in {MODEL_PATH}...")
tokenizer.save_pretrained(MODEL_PATH)
model.save_pretrained(MODEL_PATH)

print("Teuken Modell und Tokenizer erfolgreich gespeichert!") 