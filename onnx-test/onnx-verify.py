from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForCausalLM

MODEL_NAME = "rinna-neox-3.6b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ORTModelForCausalLM.from_pretrained(MODEL_NAME)

onnx_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

gen = onnx_gen("昔々あるところに")
print(gen)

gen = onnx_gen("このたびは誠に");
print(gen)

gen = onnx_gen("本日はお日柄もよく");
print(gen)

