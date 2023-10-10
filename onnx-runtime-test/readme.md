# how to build

refrence URL: [using rinna GPT on C++](https://qiita.com/shinjimorimitsu/items/b61aa693f8e2988c8590)

1. make model file for onnx runtime
for rinna/japanese-gpt2-xsmall (216MB)
```
mkdir my_onnx_gpt
optimum-cli export onnx --model rinna/japanese-gpt2-xsmall my_onnx_gpt/
```

or, for rinna/japanese-gpt-neox-small (619MB)
```
mkdir rinna-neox-small
optimum-cli export onnx --model rinna/japanese-gpt-neox-small rinna-neox-small/
```

or for rinna/japanese-gpt-neox-3.6b () (This does not work, WinRT API failed to load external data)
```
mkdir rinna-neox-3.6b
optimum-cli export onnx --model rinna/japanese-gpt-neox-3.6b rinna-neox-3.6b/
```

```
mkdir stockmark-gpt-neox-japanese-1.4b
optimum-cli export onnx --model stockmark/gpt-neox-japanese-1.4b stockmark-gpt-neox-japanese-1.4b
```


1. install sentence piece vcpkg `vcpkg install --triplet x64-windows-static`

1. open *.sln and build

