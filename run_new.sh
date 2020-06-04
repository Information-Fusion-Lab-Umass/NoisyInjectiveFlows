sbatch -p 2080ti-long --gres=gpu:1 -c 8 run_celeba512n.sh
sbatch -p 2080ti-long --gres=gpu:1 -c 8 run_celebaglown.sh

