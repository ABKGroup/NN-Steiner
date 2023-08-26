# NN-Steiner
The repo is to reproduce the results in the paper: **NN-Steiner: A Mixed Neural-algorithmic Approach for the Rectilinear Steiner Minimum Tree Problem**. 

### Environment
````
python == 3.10
torch
numpy
matplotlib
hydra-core
rtree
tqdm
scikit-learn
tensorboard
````

### Extract model

````
tar -xzvf models/m\=15_kb\=4.tar.gz
````

### Build Geosteiner
Note that the numbers reported in the paper are attained by Geosteiner compiled by cplex, but we cannot distribute anything compiled by cplex. If you have cplex, you can compile the `libgeosteiner.a` in `arora/quadtree/lib/geosteiner-5.3` by cplex.
````
cd arora/quadtree/lib
mkdir build
cd build
cmake ..
make
````

### Build RMST
````
cd arora/quadtree/rmst
mkdir build
cd build
cmake ..
make
````

### Execution
Evecution apart from the files in `evaluator` are all done through setting the configs in `conf/config.yaml` and running this command
````
python -m arora
````

### Reproduce
You can reproduce the results for NN-Steiner by running
````
bash exp/nn.sh
bash exp/ratio_flute.sh
````

### Training
To train the model, you should first produce your training data by setting the `conf/config.yaml` file. Now the training only supports GPU. If you have multiple GPUs, the training is automatically distributed on those GPUs.
````
flow: 
  - data_gen
data_gen:
  batch: <num_instances>
  num_points: <num_points>
  x_range: <x_range>
  y_range: <y_range>
  x_stdev: <x_stdev>
  y_stdev: <y_stdev>
  level: <number of levels of full quadtree, -1 if no constraint>
  output: tensor
  type: mlp
  precision: 32
  fst: 1
````
The training data will be in the `data` directory.

Then start the training by setting the `conf/config.yaml`
````
flow: 
  - train
train:
  train_set: data/<train_set>
  val_set: data/<val_set>
  test_set: data/<test_set>
  epochs: <epochs>
  max_no_update: <early stopping without improvement fot this number of evaluations>
  val_save: <true | false, whether to save the model for every evaluation or just the best model>
  lr: <learning rate>
  portal_weight: <this number + 1 is the portal weight ratio>
  batch_size: <batch_size>
  val_batch_size: <val_batch_size>
  test_batch_size: <test_batch_size>
  eval: <evaluate every this number of epoch>
````
The trained models will be in `outputs/yyyy-mm-dd/hh-mm-ss/train/model`.

### Config.yaml
````
hydra:
  job:
    chdir: true # this should be set so that the application can run
flow: # this is the parts that the application should run. The  according configurations are below
  - plot
  - data_gen
  - train
  - eval
  - solve
  - geo_exp
  - nn_exp
  - mst_exp
  - refine # refine the trees produced by FLUTE
quadtree:
  m: 15 # number of portals on one side other than the corners. Can only be 2^n - 1
  kb: 4 # maximum number of terminals in one cell
  bbox: # the bounding box of the quadtree
    width: 10000
    height: 10000
  level: -1 # minimum number of the levels in the quadtree. -1 if no constraint
model: # model related configurations
  type: mlp # only mlp is supported now
  emb_size: 16 # the output size of NNbase and NNdp is emb_size * num_portals
  hidden_size: 4096 # the size of the hidden layers in the models
  dropout: 0.1
  device: cuda:0 # the device to run the model. In training, only distributed gpus are used.
  precision: 32 # the precision of floating point in the model. Can only be 32 or 64 (32 recommended)
plot: # To plot a point set
  data: points/point2000_10000x10000-uniform-100-pt/point2000_10000x10000-uniform-0.txt
  fst: 1 # the optimality of GeoSteiner. 1 is optimal
  output: The plots to output
    - quadtree
    - golden
    - adapted
    - terminals
data_gen: # to generate training data / point sets
  seed: 42
  batch: 100 # number of cases
  num_points: 50
  width: 100 # canvas
  height: 100 # canvas
  level: -1 # number of levels for the quadtree. -1 if no constraint
  dist: uniform | normal | mix-normal | non-isotropic # The distribution to generate the points
  output: pt | tensor # tensor is for training, pt is for point sets
  type: mlp # only mlp is supported
  precision: 32 # the precision of the training data tensors
  fst: 1 # same as above
train:
  seed: 42
  train_set: data/level3_100x100-240-tensor
  val_set: data/level3_100x100-120-tensor
  test_set: data/level3_100x100-100-tensor
  checkpoint: # when you want to train from a checkpoint instead of from scratch, put the path to the checkpoint here
  epochs: 2
  max_no_update: # The training is stopped when the number of evaluations performed without model improvement exceeds the number
  val_save: true # whether to save the model other than the best model for every evaluation
  lr: 1e-4 # learning rate
  portal_weight: 15 # The portal loss weight is (portal_weight + 1):1
  batch_size: 60
  val_batch_size: 30
  test_batch_size: 100
  eval: 2 # number of training steps between two evaluations
  profile: false
eval: # this one is for only evaluating one case, and include geosteiner golden
  testcase: points/proto/point100_100x100_15_.txt
  model: models/m=7_kb=4.pt
  plot: false # whether to plot the result
  k: 10 # number of points fed into geosteiner at the end
  fst: 1 # same as above
solve: # same as eval but without geosteiner golden
  testcase: points/level3_100x100-100-pt/level3_100x100-1.txt
  model: models/m=7_kb=4.pt
  plot: false
  k: 10
  fst: 1
geo_exp: # do exp using geosteiner on a test set, you’ll get golden and snapped
  test_set: points/point100_100x100-100-pt/
  k: 10
  fst: 1
  output: exp_out/100_100x100 
nn_exp: # do exp using NN-Steiner on a test set
  test_set: points/point1000_1000x1000-100-pt/
  model: models/m=7_kb=4.pt
  k: 10
  fst: 1
  output: exp_out/1000_1000x1000
mst_exp: # do exp using mst on a test set
  test_set: points/point1000_1000x1000-100-pt/
  output: exp_out/1000_1000x1000
refine: # do refinement on tress produced by FLUTE
  point_dir: points/point50_10000x10000-100-pt/
  tree_dir: flute_results/flute50_10000x10000/
  k: 10
  fst: 1
  output: exp_out/50_10000x10000-flute
````

### Evaluators (This generates the final result)
`evaluateRatio.py`: to evaluate the performance ratio between two cases
````
python evaluateRatio.py <length1.txt> <length2.txt>
````
`evaluateSTT.py`: to see if the solution is a valid steiner tree
````
python evaluateSTT.py <point.txt> <result.txt>
````