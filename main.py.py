from fastapi import FastAPI, WebSocket
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import asyncio

app = FastAPI()

# Load model
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

async def generate_stream(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split()
    
    for token in generated_text:
        yield token + " "
        await asyncio.sleep(0.05)

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            async for token in generate_stream(f"User: {data}\nAssistant:"):
                await websocket.send_text(token)
            await websocket.send_text("[END]")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()
