# Download Datasets

## Code Summarization

You need firstly to download dataset.zip from [https://github.com/microsoft/CodeXGLUE/blob/main/Code-Text/code-to-text/dataset.zip](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Text/code-to-text/dataset.zip).
```
unzip dataset.zip
cd dataset
wget https://zenodo.org/record/7857872/files/go.zip
wget https://zenodo.org/record/7857872/files/java.zip
wget https://zenodo.org/record/7857872/files/javascript.zip
wget https://zenodo.org/record/7857872/files/php.zip
wget https://zenodo.org/record/7857872/files/python.zip
wget https://zenodo.org/record/7857872/files/ruby.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
```

## Project-specific Code Summarzation

You need to download datasets from [https://github.com/pkuserc/MPCos_ASE2022/tree/main/dataset](https://github.com/pkuserc/MPCos_ASE2022/tree/main/dataset).


