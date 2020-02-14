# DeepS3PR
This repository contains the code associated with the paper "Deep S^3PR: Simultaneous Source Separation and Phase Retrieval Using Deep Generative Models" by Chris Metzler and Gordon Wetzstein.

## Dependencies
I recommend installing dependencies via anaconda. As of 2/6/20, the following commands were sufficient:

conda install pytorch torchvision -c pytorch

conda install scipy scikit-image matplotlib pandas

pip install torch-dct

## Running the Code
With the dependencies installed, running ``python DeepS3PR_Demo.py'' should then run all the tests conducted in the paper. It will take over 12 hours to complete.

The code presently requires a GPU with CUDA support.

## Credit
The generator networks in the "Generators" folder were created using Erik Linder-Noren's Pytorch-GAN code: https://github.com/eriklindernoren/PyTorch-GAN
