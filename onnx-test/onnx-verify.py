from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForCausalLM

MODEL_NAME = "my_onnx_gpt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ORTModelForCausalLM.from_pretrained(MODEL_NAME)

onnx_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

gen = onnx_gen("昔々あるところに")
print(gen)
