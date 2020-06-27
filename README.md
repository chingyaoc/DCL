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

Train the representation encoder
```
python main.py --tau_plus = 0.1
```

Linear evaluation
```
python linear.py --model_path results/model_400.pth
```

