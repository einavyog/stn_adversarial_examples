# Spatial Transformer Network For Adversarial ExamplesGeneration:

The recent extensive use of deep neural networks (DNNs) has brought up the vulnerability o fDNNs to attacks. Adversarial Examples are inputs to the network that were designed to mislead the network and cause false results. A small change in the input can cause the network to errwith high confidence. In classification, the adversarial examples are designed to cause wrongclassification. In this work, we suggest to use a Spatial Transformer Network (STN), a learnablemodule, which explicitly allows the spatial manipulation of data, to create adversarial examples.

## Interface to train the model

``` bash
usage: main.py [-h] [--data_set DATA_SET] [--outputDir OUTPUTDIR]
               [--seed SEED] [--use_cuda USE_CUDA] [--nEpochs NEPOCHS]
               [--batchSize BATCHSIZE] [--adversarial_label ADVERSARIAL_LABEL]
               [--source_label SOURCE_LABEL] [--LR LR] [--beta BETA]
```

## Interface to use the trained STN to generate adversarial examples

``` bash
usage: generate_adversarial_examples.py [-h] [--seed SEED]
                                        [--use_cuda USE_CUDA] [--model MODEL]
                                        [--batchSize BATCHSIZE]
                                        [--adversarial_label ADVERSARIAL_LABEL]
                                        [--source_label SOURCE_LABEL]
```

## Dependencies 

1. Pytorch 0.4.0
2. Python 3.6 
2. CUDA 8.0.61
2. cuDNN 7102 

