# Deep Q-Networks for Accelerating the Training of Deep Neural Networks

> Source code to the paper [https://arxiv.org/abs/1606.01467](https://arxiv.org/abs/1606.01467)

> My current thoughts on hyperparameter optimization can be found in my [blog post](https://bigaidream.gitbooks.io/tech-blog/content/2016/thoughts-hyperparameter.html).

## Abstract and motivations

Let's take a look at the video of the training of (lower-layer) deep newtorks' weights: http://cs.nyu.edu/~yann/research/sparse/psd-anim.gif  

> Actually the video is about sparse coding, but they are similar to the training at lower-layer deep networks. 

Imagine that we are playing a weird Atari game with the above screen. The screen seems simpler than real Atari games'. This reminds us of the difference between biomedical image processing and natural image processing. In biomedical images, the objects (e.g. red blood cells) are much simpler, thus needing simpler and smaller models. This means, we can use a small-sized DQN to control the training of a larger model, e.g. a sophisticated deep convolutional neural networks. 

## Tuning learning rates on CIFAR-10 with WideResNet (50 epochs, 10 episodes)

The DQN part is modified from [DeepMind-DQN](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner).

The wide residual network part is modified from [szagoruyko-wide-resnet](https://github.com/szagoruyko/wide-residual-networks)

### Dependences
* torchnet
* optnet
* iterm

Download CIFAR_10 whitened dataset from https://yadi.sk/d/em4b0FMgrnqxy and save in `cifar_lr/dqn/datasets`

```bash
cd /cifar_lr/dqn/datasets
th split.lua; #get trainvalidata.t7
#Start tune learning rate using dqn
cd /cifar_lr
./run_gpu; 
```
The results are in `/dqn/logs/` folder. 

For setting total epochs and number of layers for wide residual net, please modify `/cifar_lr/run_gpu`

## Citation
```
@article{dqn-accelerate-dnn,
  title={Deep Q-Networks for Accelerating the Training of Deep Neural Networks},
  author={Fu, Jie and Lin, Zichuan and Chen, Danlu and Liu, Miao and Leonard, Nicholas and Feng, Jiashi and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:1606.01467},
  year={2016}
}
```

## Contact

If you have any problems or suggestions, please contact: jie.fu A~_~T u.nus.edu~~cation~~
