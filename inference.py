import torch
from customgpt import LanguageModel, encode, decode, device

# Load the model
model = LanguageModel()
model.load_state_dict(torch.load("customgpt.pth"))
model.eval()

# Function to generate a reply using the loaded model
def generate_reply(model, encode, decode, input_text, max_new_tokens=500):
    model.eval()
    context = torch.tensor(encode(input_text), dtype=torch.long, device=device).unsqueeze(0)
    generated_sequence = model.generate(context, max_new_tokens=max_new_tokens)
    reply = decode(generated_sequence[0].tolist())
    return reply

# Inference
input_text = "Gloria, come downstairs right now!"
print(generate_reply(model, encode, decode, input_text))