# NoisyInjectiveFlows
Code for Noisy Injective Flows Paper

# Workflow for experiments:
    - python plot_experiments.py --names <all names>
    - (Optional if FashionMNIST.npz doesn't exist)
        - python -c 'import datasets;datasets.save_fashion_mnist_to_samples()'
        - cd FID && python create_fashion_mnist_stats.py
    - python generate_images_for_fid.py --names <all names>
    - cd FID
    - python evaluate_fid.py --names <all names>