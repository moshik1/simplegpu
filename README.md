# How to compile:

```
mkdir build
cd build
cmake ..
make -j
```



# Download a test model
```
simplegpu> wget https://huggingface.co/ibm-granite/granite-3b-code-base/resolve/main/model-00002-of-00002.safetensors?download=true -o model.bin
```

# How to run:

```
simplegpu> build/bfloat16_test model.bin
```
