## PyTorch-Underwater-Image-Enhancement

This is the repo for "Underwater Image Enhancement based on Deep Learning and Image Formation Model"[[arXiv]](https://arxiv.org/abs/2101.00991)

The current code works with NVIDIA GPU on Ubuntu. You can do testing on CPU. 

### Requirements
    pip install -r requirements.txt

### Train the model
    $ python train.py TRAIN_RAW_IMAGE_FOLDER TRAIN_REFERENCE_IMAGE_FOLDER
### Test the model
    $ python test.py --checkpoint CHECKPOINTS_PATH
For convenience, you can run the following command to quickly see the results using the trained model reported in our paper.

    $ python test.py --checkpoint ./checkpoints/model_best_2842.pth.tar
### Citation
If you use this code in your research, please consider citing the following paper:

    @misc{chen2021underwater,
      title={Underwater Image Enhancement based on Deep Learning and Image Formation Model}, 
      author={Xuelei Chen and Pin Zhang and Lingwei Quan and Chao Yi and Cunyue Lu},
      year={2021},
      eprint={2101.00991},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
 or the Chinese journal version:
 
    [1]陈学磊,张品,权令伟,易超,鹿存跃.融合深度学习与成像模型的水下图像增强算法[J/OL].计算机工程:1-9[2021-03-15].https://doi.org/10.19678/j.issn.1000-3428.0060653.
### Acknowledgment
Sponsored by the Oceanic Interdisciplinary Program of Shanghai Jiao Tong University (project number SL2020ZD103)
