# Debiased Contrastive Learning

<p align='center'>
<img src='https://github.com/chingyaoc/DCL/blob/master/misc/fig1.png?raw=true' width='500'/>
</p>

A prominent technique for self-supervised representation learning has been to contrast semantically similar and dissimilar pairs of samples. Without access to labels, dissimilar (negative) points are typically taken to be randomly sampled datapoints, implicitly accepting that these points may, in reality, actually have the same label. Perhaps unsurprisingly, we observe that sampling negative examples from truly different labels improves performance, in a synthetic setting where labels are available. Motivated by this observation, we develop a debiased contrastive objective that corrects for the sampling of same-label datapoints, even without knowledge of the true labels.


**Debiased Contrastive Learning**
<br/>
[Ching-Yao Chuang](https://chingyaoc.github.io/), 
[Joshua Robinson](https://joshrobinson.mit.edu/), 
[Lin Yen-Chen](https://yenchenlin.me/),
[Antonio Torralba](http://web.mit.edu/torralba/www/), and
[Stefanie Jegelka](https://people.csail.mit.edu/stefje/)
<br/>


## Prerequisites
- Python 3.7 
- PyTorch 1.3.1
- PIL
- OpenCV

## Contrastive Representation Learning
We can train standard (biased) or debiased version (M=1) of [SimCLR](https://arxiv.org/abs/2002.05709) with `main.py` on STL10 dataset.

flags:
  - `--debiased`: use debiased objective (True) or standard objective (False)
  - `--tau_plus`: specify class probability
  - `--batch_size`: batch size for SimCLR

For instance, run the following command to train a debiased encoder.
```
python main.py --tau_plus = 0.1
```

## Linear evaluation
The model is evaluated by training a linear classifier after fixing the learned embedding.

path flags:
  - `--model_path`: specify the path to saved model
```
python linear.py --model_path results/model_400.pth
```

#### Pretrained Models
|          | tau_plus | Arch | Latent Dim | Batch Size  | Accuracy(%) | Download |
|----------|:---:|:----:|:---:|:---:|:---:|:---:|
|  Biased | tau_plus = 0.0 | ResNet50 | 128  | 256  | 80.15  |  [model]()|
|  Debiased |tau_plus = 0.05 | ResNet50 | 128  | 256  | 81.85  |  [model]()|
|  Debiased |tau_plus = 0.1 | ResNet50 | 128  | 256  | 84.26  |   [model]()|

## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{chuang2020debiased,
  title={Debiased Contrastive Learning},
  author={Chuang, Ching-Yao and Robinson, Joshua and Yen-Chen, Lin and Torralba, Antonio and Jegelka, Stefanie},
  journal={arXiv preprint arXiv:2007.00224},
  year={2020}
}
```
For any questions, please contact Ching-Yao Chuang (cychuang@mit.edu).

## Acknowledgements

Part of this code is inspired by [leftthomas/SimCLR](https://github.com/leftthomas/SimCLR).
