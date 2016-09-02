# Deep Q-Networks for Accelerating the Training of Deep Neural Networks

> Source code to the paper [https://arxiv.org/abs/1606.01467](https://arxiv.org/abs/1606.01467)

> My current thoughts on hyperparameter optimization can be found in my [blog post](https://bigaidream.gitbooks.io/tech-blog/content/2016/thoughts-hyperparameter.html).

> We are managing this project with ZenHub following http://devblog.edsurge.com/scrum-kanban-trello-github-issues/

## Abstract and motivations

Let's take a look at the video of the training of (lower-layer) deep newtorks' weights: http://cs.nyu.edu/~yann/research/sparse/psd-anim.gif  

> Actually the video is about sparse coding, but they are similar to the training at lower-layer deep networks. 

Imagine that we are playing a weird Atari game with the above screen. The screen seems simpler than real Atari games'. This reminds us of the difference between biomedical image processing and natural image processing. In biomedical images, the objects (e.g. red blood cells) are much simpler, thus needing simpler and smaller models. This means, we can use a small-sized DQN to contrl the training of a larger model, e.g. a ultra deep convolutional neural networks. 

However, there is one problem with this approach: when using stochastic training methods, the order of weights at very episode might change spontaneously. In analogy to applying DQNs to Atari games, it does not make sense for the Atari games to change their graphics APIs at every episode. 

In the precious versions of the arXiv paper, we added a `meta-momentum` term to constrain the training trajectory. Now we realized that this is actually quite cumbersome. We are now implement two variants:

1. Jump-start: We simply copy the weights of a CNN duing training when it achieves pretty high training accuracy (e.g. 90%) as the starting point for traning the DQN. 
2. Bag-of-filters: We first train a CNN till convergence. We treat the weights (filters) at every iteration as images and use an unsupervised learning method to build a code-book based on them. We then use this code-book to generate bag-of-words for those filters. We remove the feature learning component of the DQN, and only train the action selector. 



## TODOs
1. Binarize the weights as the states of the DQN. This will make the training of DQN even faster, [issue #22](https://github.com/bigaidream-projects/qan/issues/22)
2. Multi-agent DQNs for individual learning rate for every layer, [issue 18](https://github.com/bigaidream-projects/qan/issues/18)

## Reproduce our results on MNIST

Experimental log can be found here: https://github.com/bigaidream-experiments/qan-exp

### Dependencies
We are using Lua/Torch. The DQN component is mostly modified from [DeepMind Atari DQN](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner). 

You might need to run `install_dependencies.sh` first. 

### Tuning learning rates on MNIST
> Please refer to this old commit: https://github.com/bigaidream-projects/qan/tree/4733b2a3ca4cad4dd882a508d8fb7fd93f91615b

```bash
cd mnist_lr/;
cd mnist;
th train-on-mnist.lua; #get regression filter, save in ../save/
./run_gpu; #Start tune learning rate using dqn
#To get the test curve, run following command
cd mnist_lr/dqn/logs;
python paint_lr_episode.py;
python paint_lr_vs.py;
```

### Tuning mini-batch selection on MNIST 

> Please refer to this old commit: https://github.com/bigaidream-projects/qan/tree/4733b2a3ca4cad4dd882a508d8fb7fd93f91615b

```bash
cd mnist_minibatch;
cd mnist;
th train-on-mnist.lua; #get regression filter, save in ../save/
./run_gpu; #Start select mini-batch using dqn
#To get the test curve, run following command
cd mnist_minibatch/dqn/logs;
python paint_mini_episode.py;
python paint_mini_vs.py;
```

## Tuning learning rates on CIFAR_10 (beta version)
```bash
luarocks install torchnet
luarocks install optnet
luarocks install iterm
#Install cuDNN-5.1
wget http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-7.5-linux-x64-v5.1.tgz
tar xvf cudnn-7.5-linux-x64-v5.1.tgz
sudo cp cuda/lib64/* /usr/local/cuda/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
mkdir cifar_lr/dqn/datasets;
#Download CIFAR_10 whitened dataset from [here](https://yadi.sk/d/em4b0FMgrnqxy) and save in cifar_lr/dqn/datasets
cd cifar_lr/dqn/datasets;
th split.lua; #get trainvalidata.t7
cd ../..;
#get baseline log and save baseline weight
max_episode=1 take_action=0 savebaselineweight=1 output_file='logs/torchnet_test_baseline.log' ./run_gpu
./run_gpu; #Start tune learning rate using dqn
#To get the test curve, run following command
cd cifar_lr/dqn/logs;
python paint.py;
#and check the acc.pdf
```

### Different Settings
1. GPU device can be set in `run_gpu` where `gpu=0`
2. Learning rate can be set in `/ataricifar/dqn/cnnGameEnv.lua`, in the `step` function. 
3. When to stop doing regression is in `/ataricifar/dqn/cnnGameEnv/lua`, in line 250

---

## FAQ
1. `Q:` Do the actions change over time? Do they converge to something like the alternated classes with uniform sampling heuristics that is always used in CNNs? 
`A:` From what we observed, the actions do change over time. But the pattern is much more complex than uniform sampling heuristics. The visualization plot can be found [here](https://github.com/bigaidream-experiments/qan-exp/blob/master/batchvisualization/20160724/batchvisual.pdf)


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
