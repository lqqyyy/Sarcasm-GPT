import json
import os
import sys
import argparse


import torch
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from utils.prompter import Prompter
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn


app = FastAPI()

class InferenceInput(BaseModel):
    instruction: str


global_model = None
global_tokenizer = None
global_prompter = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def load_model(base_model, lora_weights, load_8bit):
    print(base_model)
    print(lora_weights)
    print(load_8bit)

    global device

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    print(device)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            # load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    if not load_8bit:
        model.half()

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return model, tokenizer, Prompter("")

def evaluate(model, tokenizer, prompter, instruction):
    prompt = prompter.generate_prompt(instruction, "")

    # print("==============prompt==============")
    # print(prompt)
    # print("==================================")


    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.75,
            # temperature=0.1,
            temperature=0.6,
            top_k=40,
            num_beams=4,
            return_dict_in_generate=True,
            output_scores=True,
        )

    output = tokenizer.decode(generation_output.sequences[0])

    print("==============output==============")
    print(output)
    print("==================================\n\n\n")

    return prompter.get_response(output)

@app.post("/stream/")
async def create_item(request: Request):
    try:
        json_post_raw = await request.json()
        json_post = json.dumps(json_post_raw)
        json_post_list = json.loads(json_post)
        prompt = json_post_list.get('prompt')
        history = json_post_list.get('history')

        response = evaluate(global_model, global_tokenizer, global_prompter, prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--lora_weights', type=str)
    parser.add_argument('--load_8bit', action='')
    args = parser.parse_args()

    global_model, global_tokenizer, global_prompter = load_model(args.base_model, args.lora_weights, args.load_8bit)


    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
