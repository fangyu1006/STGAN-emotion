# STGAN 


- **Prerequisites**
    - Tensorflow (r1.4 - r1.12 should work fine)
    - Python 3.x with matplotlib, numpy and scipy

- **Dataset**
    - [CelebA](http://openaccess.thecvf.com/content_iccv_2015/papers/Liu_Deep_Learning_Face_ICCV_2015_paper.pdf) dataset (Find more details from the [project page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))
        - [Images](https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM) should be placed in ***DATAROOT/img_align_celeba/\*.jpg***
        - [Attribute labels](https://drive.google.com/open?id=0B7EVK8r0v71pblRyaVFSWGxPY0U) should be placed in ***DATAROOT/list_attr_celeba.txt***
        - If google drive is unreachable, you can get the data from [Baidu Cloud](http://pan.baidu.com/s/1eSNpdRG)
    - We follow the settings of AttGAN, kindly refer to [AttGAN](https://github.com/LynnHo/AttGAN-Tensorflow) for more dataset preparation details

## Quick Start

Exemplar commands are listed here for a quick start.

### Training

- for 128x128 images

    ```console
    python train.py --experiment_name 128
    ```

- for 384x384 images (please prepare data according to [HD-CelebA](https://github.com/LynnHo/HD-CelebA-Cropper))

    ```console
    python train.py --experiment_name 384 --img_size 384 --enc_dim 48 --dec_dim 48 --dis_dim 48 --dis_fc_dim 512 --n_sample 24 --use_cropped_img
    ```

### Testing

- Example of testing ***single*** attribute

    ```console
    python test.py --experiment_name 128 [--test_int 1.0]
    ```

- Example of testing ***multiple*** attributes

    ```console
    python test.py --experiment_name 128 --test_atts Pale_Skin Male [--test_ints 1.0 1.0]
    ```

- Example of ***attribute intensity control***

    ```console
    python test.py --experiment_name 128 --test_slide --test_att Male [--test_int_min -1.0 --test_int_max 1.0 --n_slide 10]
    ```

The arguments in `[]` are optional with a default value.

### View Images

You can use `show_image.py` to show the generated images, the code has been tested on Windows 10 and Ubuntu 16.04 (python 3.6). If you want to change the width of the buttons in the bottom, you can change `width` parameter in the 160th line. the '+++' and '---' on the button indicate that the above image is modified to 'add' or 'remove' the attribute. Note that you should specify the path of the attribute file (`list_attr_celeba.txt`) of CelebA in the 82nd line.

### NOTE:

- You should give the path of the data by adding `--dataroot DATAROOT`;
- You can specify which GPU to use by adding `--gpu GPU`, e.g., `--gpu 0`;
- You can specify which image(s) to test by adding `--img num` (e.g., `--img 182638`, `--img 200000 200001 200002`), where the number should be no larger than 202599 and is suggested to be no smaller than 182638 as our test set starts at 182638.png.
- You can modify the model by using following arguments
    - `--label`: 'diff'(default) for difference attribute vector, 'target' for target attribute vector
    - `--stu_norm`: 'none'(default), 'bn' or 'in' for adding no/batch/instance normalization in STUs
    - `--mode`: 'wgan'(default), 'lsgan' or 'dcgan' for differenct GAN losses
    - More arguments please refer to [train.py](./train.py)

### AttGAN

- Train with AttGAN model by

    ```console
    python train.py --experiment_name attgan_128 --use_stu false --shortcut_layers 1 --inject_layers 1
    ```

## Acknowledgement
The code is built upon [STGAN](https://github.com/csmliu/STGAN), thanks for their excellent work!
