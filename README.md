# Provably Good Batch Reinforcement Learning Without Great Exploration

Code for the algorithm Pessimistic Q-learning (PQL) in our paper [Provably Good Batch Reinforcement Learning Without Great Exploration](https://arxiv.org/abs/2007.08202). Pessimistic Q-learning (PQL) is a deep batch reinforcement learning algorithm that is based on the pessimistic Q iteration (PQI) and policy iteration (PPI) algorithm with provable guarantees in the paper. Please see the paper for more details.



This code is built on top of the [implementation of BCQ](https://github.com/sfujim/BCQ) and [implementation of BEAR](https://github.com/aviralkumar2907/BEAR). The method is tested in 3 [MuJoCo](http://www.mujoco.org/) continuous control tasks in the [D4RL](https://github.com/rail-berkeley/d4rl) benchmark. Thus it requires to install MuJoCo and D4RL first to run this code. 



In order to run the experiment of PQL in the paper, first a script need to load the dataset from D4RL and transform it to our data loader's form:
```
python load_dataset.py
```

Then please use:
```
python train.py --env=Hopper-v2 --dataset=d4rl-hopper-medium-v0
python train.py --env=HalfCheetah-v2 --dataset=d4rl-halfcheetah-medium-v0 --ql_noise=0.0 
python train.py --env=Walker2d-v2 --dataset=d4rl-walker2d-medium-v0 --ql_noise=0.0 
``` 

If you use this code in your research, please cite our paper:
### Citation

```
@misc{liu2020provably,
      title={Provably Good Batch Reinforcement Learning Without Great Exploration}, 
      author={Yao Liu and Adith Swaminathan and Alekh Agarwal and Emma Brunskill},
      year={2020},
      eprint={2007.08202},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```