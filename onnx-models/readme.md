# how to build

refrence URL: [using rinna GPT on C++](https://qiita.com/shinjimorimitsu/items/b61aa693f8e2988c8590)

1. install onnx exporter

```
pip install optimum[exporters]
```

1. convert rinna/japanese-gpt2-xsmall (216MB)
```
mkdir rinna-gpt2-xsmall
optimum-cli export onnx --model rinna/japanese-gpt2-xsmall rinna-gpt2-xsmall/
```

1. rinna/japanese-gpt2-small
```
mkdir rinna-japanese-gpt2-small
optimum-cli export onnx --model rinna/japanese-gpt2-small rinna-japanese-gpt2-small/
```


1. rinna/japanese-gpt-neox-small (619MB)
```
mkdir rinna-neox-small
optimum-cli export onnx --model rinna/japanese-gpt-neox-small rinna-neox-small/
```

1. rinna/japanese-gpt-neox-3.6b () (This does not work, WinRT API failed to load external data)
```
mkdir rinna-neox-3.6b
optimum-cli export onnx --model rinna/japanese-gpt-neox-3.6b rinna-neox-3.6b/
```

```
mkdir rinna-bilingual-gpt-neox-4b
optimum-cli export onnx --model rinna/bilingual-gpt-neox-4b rinna-bilingual-gpt-neox-4b/
```

1. stockmark/gpt-neox-japanese-1.4b
```
mkdir stockmark-neox-1.4b
optimum-cli export onnx --model stockmark/gpt-neox-japanese-1.4b stockmark-neox-1.4b/
```


1. install sentence piece vcpkg `vcpkg install --triplet x64-windows-static`

1. open *.sln and build

