from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForCausalLM

# MODEL_NAME = "rinna-neox-3.6b"
MODEL_NAME = "stockmark-gpt-neox-japanese-1.4b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ORTModelForCausalLM.from_pretrained(MODEL_NAME)

onnx_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

gen = onnx_gen("私の姉は陽子で、旦那の妹は葉子です。姉の名前は")
print(gen)

gen = onnx_gen("私の姉は葉子で、旦那の妹は洋子です。姉の名前は")
print(gen)

gen = onnx_gen("私の姉は陽子で、旦那の妹は葉子です。義理の妹の名前は")
print(gen)

gen = onnx_gen("私の姉は葉子で、旦那の妹は陽子です。義理の妹の名前は")
print(gen)

gen = onnx_gen("旦那の妹は葉子で、私の姉は陽子です。義理の妹の名前は")
print(gen)

gen = onnx_gen("旦那の妹は洋子で、私の姉は葉子です。義理の妹の名前は")
print(gen)



# gen = onnx_gen("昔々あるところに")
# print(gen)
# 
# gen = onnx_gen("このたびは誠に");
# print(gen)
# 
# gen = onnx_gen("本日はお日柄もよく");
# print(gen)

