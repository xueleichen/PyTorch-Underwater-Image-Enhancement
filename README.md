## PyTorch-Underwater-Image-Enhancement

This is the repo for "Underwater Image Enhancement based on Deep Learning and Image Formation Model"[[arXiv]](https://arxiv.org/abs/2101.00991)

The current code works with NVIDIA GPU on Ubuntu. I am not sure if it works on other plateforms or with CPU. 

### Requirements
    Pytorch==1.6.0
    pillow==7.2.0

### Train the model
    $ python train.py TRAIN_RAW_IMAGE_FOLDER TRAIN_REFERENCE_IMAGE_FOLDER
### Test the model
    $ python test.py CHECKPOINTS_PATH TEST_RAW_IMAGE_FOLDER
For convenience, you can run the following command to quickly see the results using the trained model reported in our paper.

    $ python test.py ./checkpoints/model_best_2842.pth.tar ./test_img/
### Citation
If you use this code in your research, please cite the following paper:

    @misc{chen2021underwater,
      title={Underwater Image Enhancement based on Deep Learning and Image Formation Model}, 
      author={Xuelei Chen and Lingwei Quan and Chao Yi and Cunyue Lu},
      year={2021},
      eprint={2101.00991},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
    }
