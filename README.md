# NoisyInjectiveFlows
Code for Noisy Injective Flows Paper

## Dependencies
See `requirements.txt` for required pip packages, or use the following to download all dependencies 
all dependencies:
```bash
pip install -r requirements.txt
```

## Data
fashion-mnist, cifar10, and celebA were used in the paper. fashion-mnist and cifar10 are downloaded automatically 
using the dataloader found in datasets.py. CelebA must be downloaded seperately from https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8

## Usage

### Training
To train a model, first create a yaml file for the model architecture. All model's used in the paper have an existing yaml file.
Once a yaml file exists, run the following:
```
python train_runner.py --name yamlfilename
```
### Running Experiments
To run experiments, run the following:
```
python plot_experiments.py --names model_name1 --experiment_root 'Experiments'
```
additional commandline arguments for running each experiment is as follows:
--compare_baseline_samples
--compare_vertical
--compare_samples
--compare_full_samples
--reconstructions 
--vary_t
--vary_s
--compare_t
--interpolations
--best_s_for_nll
--nll_comparison
--save_embedding
--plot_embeddings
To run the plot_embeddings experiment, you must first run save_embedding seperately for the two model you wish to compare, for example:
```
python plot_experiments.py --name celeba_64n --experiment_root 'Experiments' --save_embedding
python plot_experiments.py --name celeba_glown --experiment_root 'Experiments' --save_embedding
```
To run plot_embeddings, on these two models, you would then run:
```
python plot_experiments.py --name celeba_64n celeba_glown --experiment_root 'Experiments' --plot_embeddings
```

### Running FID Scores
To run the FID scores, you need a valid version of tensorflow (we used tensorflow-gpu version 1.15.2).  This version is only supported in Python 3.7.  There is a Dockerfile in the FID folder that can be used to create an environment to run the FID code.

To generate the FID scores, you must first run generate_images_for_fid.py on existing experiments.  This will create a folder of images that will be fed into the FID code.
```
python generate_images_for_fid.py --names <experiment names>
```

Next, go into the FID folder and, using the correct environment, run evaluate_fid.py
```
python evaluate_fid.py --names <experiment names>
```

