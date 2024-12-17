
# A toy FizzBuzz model trainer

[main.py](main.py) - (obsolete) uses python multiprocessing to run multiple training sessions in parallel.

[main_parallel.py](main_parallel.py) - (run this instead) uses `torch.spawn()` to do a not-much-better job of running ranks in parallel. 


The program includes a class that contains all the starting values.  The [Hyperparameters](hyperparameters.py) class is passed around to all the other classes that might reference it.  Since it's injected just about everywhere, you can use it to parameterize any magic numbers.

The [PerturbRule](perturbations.py) class is a mechanism to programatically change some hyperparameters across several runs.  You define the rules in the `if __name__ == '__main__':` section of [main_parallel.py](main_parallel.py)

[animate.py](animate.py) can generate animations by reloading models and plotting layers, to visualize training progress.  You do this _after_ the training run, using the checkpoints in your `results/#rank#/checkpoints/` folder.  If you didn't enable `save_checkpoints`, you will not have anything to animate. 

### outputs
The program creates a folder named `results/` and produces output in a timestamp-named folder, for each run.
- `results/#rank#/hyperpermerters.json` - records the set of hyperparameters that this rank was started with.
- `results/#rank#/model.pth` - the last checkpoint, if the model achieved perfection.* 
- `results/#rank#/checkpoints/model_000000.pth` - if you set `save_checkpoints = True`, then every epoch will save its checkpoint to a numbered file. These artifacts are used by [animate.py](animate.py)


While fiddling around with the parameters, I got a perfect* model in 6 epochs.  

hidden_dim @ 410 or 430 and</br>
input_duplicates @ 57 or 63</br>
have the best results for some reason.


<sub>*perfect means that it predicts correct output for all FizzBuzz inputs in range(101)</sub>
### example training session
```bash
(fbai) mark@four:~/prog/fbai$ python main_parallel.py
[   0] hidden_dim: 430.000, 430.000, ..., 430.000
[   0] initial_learning_rate: 0.010, 0.010, ..., 0.010
[   0] input_duplicates: 63.000, 63.000, ..., 63.000
[   0] initial_learning_rate = 0.0097, input_duplicates = 63, hidden_dim = 430
[   1] initial_learning_rate = 0.0097, input_duplicates = 63, hidden_dim = 430
[   0]Epoch     0 t: 0.0101371 v: 0.7424107 c: 145/185   lr:0.00970
[   0]Epoch     1 t: 0.0000746 v: 0.4787807 c: 159/185   lr:0.00970
[   0]Epoch     2 t: 0.0000076 v: 0.3310000 c: 167/185   lr:0.00970
[   0]Epoch     3 t: 0.0000035 v: 0.2319925 c: 169/185   lr:0.00970
[   0]Epoch     4 t: 0.0000030 v: 0.1585426 c: 174/185   lr:0.00970
[   0]Epoch     5 t: 0.0000028 v: 0.1631434 c: 175/185 * lr:0.00970 p:  99 / 100
[   0]Epoch     6 t: 0.0000027 v: 0.1045407 c: 179/185 * lr:0.00970 p: 100 / 100

Final test evaluation
Epoch     6 val: 100 / 100
[   0] initial_learning_rate = 0.0097, input_duplicates = 63, hidden_dim = 430
WIN!... Model saved
```