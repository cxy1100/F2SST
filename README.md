# F2SST
Code for **F2SST: Frequency-to-Spatial Semantic Transfer for Few-Shot Image Classification.**

## Datasets
We use four standard few-shot learning benchmarks. Please prepare the datasets as follows:

miniImageNet:

Obtain the dataset using https://github.com/yaoyao-liu/mini-imagenet-tools.

Place it in ./datasets/mini/.

tieredImageNet:

Obtain the dataset using https://github.com/yaoyao-liu/tiered-imagenet-tools

Place it in ./datasets/tiered/.

CIFAR-FS:

Run the download script: bash [download_cifar_fs.sh](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_cifar_fs.sh)

Place it in ./datasets/cifar_fs/.

FC100:

Run the download script: bash [download_fc100.sh](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_fc100.sh)

Place it in ./datasets/FC100/.

Ensure the directory structure is:

```plaintext
your-project-name/
├── datasets/
│   ├── mini/          # Place miniImageNet here
│   ├── tiered/        # Place tieredImageNet here
│   ├── cifar_fs/      # Place CIFAR-FS here
│   └── FC100/         # Place FC100 here
├── ...
```

## Backbone Pre-training
We use a pre-trained ViT-small backbone. Please follow the instructions in FewTURE to obtain the model.

Save the pre-trained weights to the directory of the specific dataset you are using:

./initialization/{dataset}/

Note: Please replace {dataset} with the actual dataset name (e.g., mini, tiered, cifarfs, or fc100).

## Acknowledgment
We thank the following repos providing helpful components/functions in our work.

[FEAT](https://github.com/Sha-Lab/FEAT), [FewTURE](https://github.com/mrkshllr/FewTURE), [CPEA](https://github.com/FushengHao/CPEA)
