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

```
@article{DBLP:journals/corr/abs-2102-04664,
  author    = {Shuai Lu and
               Daya Guo and
               Shuo Ren and
               Junjie Huang and
               Alexey Svyatkovskiy and
               Ambrosio Blanco and
               Colin B. Clement and
               Dawn Drain and
               Daxin Jiang and
               Duyu Tang and
               Ge Li and
               Lidong Zhou and
               Linjun Shou and
               Long Zhou and
               Michele Tufano and
               Ming Gong and
               Ming Zhou and
               Nan Duan and
               Neel Sundaresan and
               Shao Kun Deng and
               Shengyu Fu and
               Shujie Liu},
  title     = {CodeXGLUE: {A} Machine Learning Benchmark Dataset for Code Understanding
               and Generation},
  journal   = {CoRR},
  volume    = {abs/2102.04664},
  year      = {2021}
}
@article{Rui2022MPCos,
  title={Low-Resources Project-Specific Code Summarization},
  author={Rui Xie, Tianxiang Hu, Wei Ye, Shikun Zhang},
  journal={37th {IEEE/ACM} International Conference on Automated Software Engineering,
               {ASE} 2022},
  year={2022}
}
```
