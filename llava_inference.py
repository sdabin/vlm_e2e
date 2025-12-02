import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=False,  # accelerate 없이 사용
).to(0)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
processor = AutoProcessor.from_pretrained(model_id, tokenizer=tokenizer)
# processor = AutoProcessor.from_pretrained(model_id)
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What are these?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "082.jpg"
raw_image = Image.open("082.jpg")
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
input_ids = inputs["input_ids"]
output = model.generate(**inputs, max_new_tokens=7, do_sample=False)
generated_tokens = output[:, input_ids.shape[1]:]
print(processor.decode(output[0][2:], skip_special_tokens=True))
import pdb
pdb.set_trace()
# def load_llava(device):
#     model_id = "llava-hf/llava-1.5-7b-hf"
#     model = LlavaForConditionalGeneration.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16, 
#         low_cpu_mem_usage=False,  # accelerate 없이 사용
#     ).to(device)

#     tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
#     processor = AutoProcessor.from_pretrained(model_id, tokenizer=tokenizer)
#     return model, processor

# def infer_llava(model, processor, raw_image, text, max_new_token):
#     conversation = [
#         {

#         "role": "user",
#         "content": [
#             {"type": "text", "text": text},
#             {"type": "image"},
#             ],
#         },
#     ]
#     prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#     # raw_image = Image.open(img_path)
#     inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
#     input_ids = inputs["input_ids"]
#     output = model.generate(**inputs, max_new_tokens=max_new_token, do_sample=False)
#     generated_tokens = output[:, input_ids.shape[1]:]

#     return generated_tokens

# img = Image.open("082.jpg")
# model, processor = load_llava(0)
# output = infer_llava(model, processor,img, "describe", 8)
