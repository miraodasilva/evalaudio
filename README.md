# evalaudio
This repo allows you to easily compare audio samples using objective evaluation metrics, reproducing the evaluation procedure presented in our [work](https://sites.google.com/view/video-to-speech). It contains two ASR models: one for GRID and one for LRW (taken from [this repo](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks)).

## Software requirements
Python 3.7+

Pip requirements under [requirements.txt](requirements.txt), and ctcdecode==0.4, which can be installed from this [repository](https://github.com/parlance/ctcdecode) as such:
```
git clone https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
```

## Setup
First, extract GRID annotations:
```
cd WER
unzip annotations.zip
```

Second, extract LRW ckpt:
```
cd WER/LRW
unzip model_best.zip
```

Now, we are ready for evaluation.

You should place the real and generated audio in two symmetrical folders. for GRID, the directory structure should be:
```
real_grid
├── s1 
|   ├── bbaf2n.wav
|   └──...
├──s2
|   ├──bwaf3s.wav
|   └──...
└── ...

generated_grid
├──s1 
|   ├──bbaf2n.wav
|   └──...
├──s2
|   ├──bwaf3s.wav
|   └──...
└── ...
```
For LRW it should be:
```
real_lrw
├── ALREADY
|    └── train OR val OR test
|        ├── ALREADY_00001.wav
|        └── ...
├── ABOUT
|   └── train OR val ORtest
|       ├── ABOUT_00001.wav
|       └── ...
└── ...

generated_lrw
├── ALREADY
|   └── train OR val OR test
|       ├── ALREADY_00001.wav
|       └── ...
├── ABOUT
|   └── train OR val ORtest
|       ├── ABOUT_00001.wav
|       └── ...
└── ...
```

.npz and .wav can both  be used as audio formats for both folders. 16khz sampling rate is assumed for real audio and for generated audio. For generated audio 8 khz and 50khz are also compatible by using --resample_8khz or --resample_50khz.

## Usage
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_folder.py --real_folder ./real_grid --fake_folder ./generated_grid --dataset grid
```
(use --dataset lrw for LRW)
