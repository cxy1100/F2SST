# F2SST
Code for F2SST: Frequency-to-Spatial Semantic Transfer for Few-Shot Image Classification

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
│   ├── cifarfs/       # Place CIFAR-FS here
│   └── fc100/         # Place FC100 here
├── ...
```
