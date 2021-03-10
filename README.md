# The basement for the experiments for hyperparameter optimization (HPO) on SIGNATE lemon competition

## Usage
1. create virtual environmental
2. download datasets from SIGNATE
3. write shell file
4. training

### 1. create virtual environmental
アナコンダもしくはミニコンダを使う．
想定環境: miniconda3-4.7.10

```
conda env create -f env.yaml
conda activate Lemon
```

### 2. download datasets from SIGNATE
データをダウンロードする前に、トークンを作成する必要がある．

1. アカウント設定画面の"API Token"の”作成”をクリック
2. 取得したAPI Tokenを`~/.signate`直下に配置．なければ作成する

`datasets`ディレクトリに必要なデータをダウンロードする．
```
cd ./obj_functions/machine_learning_utils/datasets
signate list # confirm competition list
signate files --competition-id=431 # display download files
signate download --competition-id=431
unzip train_images.zip
```
or
```
chmod u+x ./get_datasets.sh
bash ./get_datasets.sh
```

### 3. write shell file
`bash`ディレクトリ内にshellファイルを作成する．
```
python nm_main.py -fuc wrn -ini 1 -eva 50 -dat lemon -cls 4 -eexp ResNet50 -exp 0 -res 1 -epch 200 -mtra ResNet50 -opt sgd -gcam 1
```

パラメータの説明
- `fuc` (str): The name of callable you want to optimize.
- `ini` (int): The number of initial samplings.
- `eva` (int): The maximum number of evaluations in an experiment. If eva = 100, 100 configurations will be evaluated.
- `dat` (str): The name of dataset.
- `epch` (int): The number of epochs
- `mtra` (str): The model of Transfer, default is ResNet50
- `opt` (str): The optimizer of Training, only availalbe SGD, ADAM, SAM, default is SGD
- `gcam` (int): Whether turn on GradCam or not, if 1 is using GradCam, default is 0

### 4. training
作成したshellファイルを指定して実行する．
`history`ディレクトリ内に学習・最適化結果と学習済モデルが保存される．

example) 
```
python nm_main.py -fuc wrn -ini 1 -eva 50 -dat lemon -cls 4 -eexp ResNet50 -exp 0 -res 1 -epch 2 -mtra ResNet50 -opt sgd
```
or
```
chmod u+x ./submit_transfer.sh
bash ./submit_transfer.sh
```


### Detail

## Requirements
・python3.7 (3.7.4)

・ConfigSpace 0.4.10 [ (github)](https://github.com/automl/ConfigSpace)

・pybnn 0.0.5 [ (github)](https://github.com/automl/pybnn)

・Pytorch 1.2.0 [ (github)](https://github.com/pytorch/pytorch)

・botorch 0.1.3 [ (github)](https://github.com/pytorch/botorch)

## Implementation
An easy example of `main.py`.
Note that the optimization is always minimization;
Therefore, users have to set the output multiplied by -1 when hoping to maximize.

```
import utils
import optimizer


if __name__ == '__main__':
    opt_requirements, experimental_settings = utils.parse_requirements()
    hp_utils = utils.HyperparameterUtilities(experimental_settings)
    opt = optimizer.SingleTaskGPBO(hp_utils, opt_requirements, experimental_settings)
    best_conf, best_performance = opt.optimize()
```

Run from termianl by (one example):

```
python main.py -fuc sphere -dim 2 -par 1 -ini 10 -exp 0 -eva 100 -res 0
```

where all the arguments are integer.

### fuc (Required)
The name of callable you want to optimize.

### dim (optional: Default is None)
The dimension of input space.
Only for benchmark functions.

### par (optional: Default is 1)
The number of parallel computer resources such as GPU or CPU.

### ini (Required)
The number of initial samplings.

### exp (optional: Default is 0)
The index of experiments.
Used only for indexing of experiments.

### eva (optional: Default is 100)
The maximum number of evaluations in an experiment.
If eva = 100, 100 configurations will be evaluated.

### res (optional: Default is 0(=False))
Whether restarting an experiment based on the previous experiment.
If 0, will remove the previous log files after you choose "y" at the caution.

### seed  (optional: Default is None)
The number to specify the random seed.

### veb (optional: Default is 1)
Whether print the result or not. If 0, not print.

### fre (optional: Default is 1)
Every print_freq iteration, the result will be printed.

### che (optional: Default is 1)
If asking when removing files or not at the initialization.

### dat (supervised learning)
The name of dataset.

### cls (supervised learning)
The number of classes on a given task.

### img (optional: Default is None)
The pixel size of training data.

### sub (optional: Default is None)
How many percentages of training data to use in training (Must be between 0. and 1.).

### test (optional: Default is 0)
If using validation set or test set.
If 1, using test dataset.

### defa (optional: Default is 0)
Evaluating the default configuration or not.
If 1, evaluate the default one.

### altr (optional: Default is 0)
Evaluating the default configuration or not.
If 1, evaluate the default one.

### cuda (optional: list of 0 to # of GPUs)
Which CUDA devices you use in the experiment.
Specify the single or multiple number(s).

### altr (optional: Default is 0)
If using all the training data or not.
If 1, using all the data.

### tra (required for transfer learning: list of path)
The path of the information to transfer.
The path is like "optname/func/name/experiments_number"

## Optimizer
You can add whatever optimizers you would like to use in this basement.
By inheriting the `BaseOptimizer` object, you can use basic function needed to start HPO.
A small example follows below:

```
from optimizer.base_optimizer import BaseOptimizer


class OptName(BaseOptimizer):
    def __init__(self,
                 hp_utils,  # hyperparameter utility object
                 opt_requirements,  # the variables obtained from parser
                 experimental_settings,  # the variables obtained from parser
                 **kwargs
                 ):

        # inheritance
        super().__init__(hp_utils, opt_requirements, experimental_settings)

        # optimizer in BaseOptimizer object
        self.opt = self.sample

    def sample(self):
        """
        some procedures and finally returns a hyperparameter configuration
        this hyperparameter configuration must be on usual scales.
        """

        return hp_conf
```

## Hyperparameters of Objective Functions
Describe the details of hyperparameters in `params.json`.

### 1. First key (The name of an objective function)
The name of objective function and it corresponds to the name of objective function callable.

### 2. y_names
The names of the measurements of hyperparameter configurations

### 3. y_upper_bounds
The upper bounds of each objective function. if there is no description, treated as 1.0e+8.

### 4. in_fmt
The format of input for the objective function. Either 'list' or 'dict'.

### 5. config
The information related to the hyperparameters.

#### 5-1. the name of each hyperparameter
Used when recording the hyperparameter configurations.

#### 5-2. lower, upper
The lower and upper bound of the hyperparameter.
Required only for float and integer parameters.

#### 5-3. dist (required anytime)
The distribution of the hyperparameter.
Either 'u' (uniform) or 'c' (categorical).

#### 5-4. q
The quantization parameter of a hyperparameter.
If omited, q is going to be None.
Either any float or integer value or 'None'.

#### 5-5. log
If searching on a log-scale space or not.
If 'True', on a log scale.
If omited or 'False', on a linear scale.

#### 5-6. var_type (required anytime)
The type of a hyperparameter.
Either 'int' or 'float' or 'str' or 'bool'.

#### 5-7. choices (required only if dist is 'c' (categorical) )
The choices of categorical parameters.
Have to be given by a list.

#### 5-8. ignore (optional: "True" or "False")
Whether ignoring the hyperparameter or not.

An example follows below.

```
{
    "sphere": {
      "y_names": ["loss"],
      "in_fmt": "list",
      "config": {
            "x": {
                "lower": -5.0, "upper": 5.0,
                "dist": "u", "var_type": "float"
            }
        }
    },
    "cnn": {
      "y_names": ["error", "cross_entropy"],
      "y_upper_bounds": [1.0, 1.0e+8],
      "in_fmt": "dict",
      "config": {
            "batch_size": {
                "lower": 32, "upper": 256,
                "dist": "u", "log": "True",
                "var_type": "int"
            },
            "lr": {
                "lower": 5.0e-3, "upper": 5.0e-1,
                "dist": "u", "log": "True",
                "var_type": "float"
            },
            "momentum": {
                "lower": 0.8, "upper": 1.0,
                "dist": "u", "q": 0.1,
                "log": "False", "var_type": "float"
            },
            "nesterov": {
                "dist": "c", "choices": [True, False],
                "var_type": "bool", "ignore": "True"
            }
        }
    }
}
```

## Objective Functions
The target objective function in an experiment.
This function must receive the `gpu_id`, `hp_conf`, `save_path`, and `experimental_settings` from `BaseOptimizer` object and return the performance by a dictionary format.
An example of (`obj_functions/benchmarks/sphere.py`) follows below.


```
import numpy as np

"""
Parameters
----------
experimental_settings: NamedTuple
    The NamedTuple of experimental settings.

Returns
-------
_imp: callable
"""

def f(experimental_settings):
    def _imp():
        """
        Parameters
        ----------
        hp_conf: 1d list of hyperparameter value
            [the index for a hyperparameter]
        gpu_id: int
            the index of a visible GPU
        save_path: str
            The path to record training.

        Returns
        -------
        ys: dict
            keys are the name of performance measurements.
            values are the corresponding performance.
        """

        return {"loss": (np.array(hp_conf) ** 2).sum()}

    return _imp
```

Also, the keys and corresponding values of `experimental_settings` are as follows:

### func_name: str
The name of objective function.

### dim: int
The dimension of input space.
Only for benchmark functions.

### dataset_name: str
The name of dataset.

### n_cls: int
The number of classes on a given task.

### image_size: int
The pixel size of training data.

### data_frac: float
How many percentages of training data to use in training (Must be between 0. and 1.).

### biased_cls: list of float
The size of this list must be the same as n_cls.
The i-th element of this list is the percentages of training data to use in learning.

### test: bool
If using validation set or test set.
If True, using test set.

### hpo_basement
This pipeline is forked from https://github.com/nabenabe0928/hpo_basement

### optimizer
・SAM [ (github)](https://github.com/davda54/sam)

### Grad Cam
・Grad Cam [ (github)](https://github.com/jacobgil/pytorch-grad-cam)
