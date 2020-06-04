sbatch -p 2080ti-short --gres=gpu:1 run_embeddings128.sh
sbatch -p 2080ti-short --gres=gpu:1 run_embeddings256.sh
sbatch -p 2080ti-short --gres=gpu:1 run_embeddings512.sh
