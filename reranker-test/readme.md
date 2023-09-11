# how to build

refrence URL: [using rinna GPT on C++](https://qiita.com/shinjimorimitsu/items/b61aa693f8e2988c8590)

1. make model file for onnx runtime
```
mkdir my_onnx_gpt
optimum-cli export onnx --model rinna/japanese-gpt2-xsmall my_onnx_gpt/
```

1. install sentence piece vcpkg `vcpkg install --triplet x64-windows-static`

1. open *.sln and build

