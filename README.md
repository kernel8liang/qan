# Deep Q-Networks for Accelerating the Training of Deep Neural Networks

> Source code to the paper [https://arxiv.org/abs/1606.01467](https://arxiv.org/abs/1606.01467)

> My current thoughts on hyperparameter optimization can be found in my [blog post](https://bigaidream.gitbooks.io/tech-blog/content/2016/thoughts-hyperparameter.html).

> We are managing this project with ZenHub following http://devblog.edsurge.com/scrum-kanban-trello-github-issues/

## Abstract and motivations

Let's take a look at the video of the training of (lower-layer) deep newtorks' weights: http://cs.nyu.edu/~yann/research/sparse/psd-anim.gif  

> Actually the video is about sparse coding, but they are similar to the training at lower-layer deep networks. 

Imagine that we are playing a weird Atari game with the above screen. The screen seems simpler than real Atari games'. This reminds us of the difference between biomedical image processing and natural image processing. In biomedical images, the objects (e.g. red blood cells) are much simpler, thus needing simpler and smaller models. This means, we can use a small-sized DQN to contrl the training of a larger model, e.g. a ultra deep convolutional neural networks. 

However, there is one problem with this approach: when using stochastic training methods, the order of weights at very episode might change spontaneously. In analogy to applying DQNs to Atari games, it does not make sense for the Atari games to change their graphics APIs at every episode. 

## Tuning learning rates on CIFAR_10 with WideResNet (10 epochs, 10 episodes)
```bash
luarocks install torchnet
luarocks install optnet
luarocks install iterm

mkdir cifar_lr/dqn/datasets;
#Download CIFAR_10 whitened dataset from [here](https://yadi.sk/d/em4b0FMgrnqxy) and save in cifar_lr/dqn/datasets
cd cifar_lr/dqn/datasets;
th split.lua; #get trainvalidata.t7
cd ../..;
/run_gpu
./run_gpu; #Start tune learning rate using dqn

```

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
