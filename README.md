# Accelerating Bayesian Optimization for Protein Design via Evolutionary Acquisition Optimization

_CSCI 5980 Final Project (University of Minnesota, Spring 2026)_

_Special Topics: AI for Sequential Decision Making with Prof. Aryan Deshwal_

Pranshu Panda and Drew Gjerstad

## Contents
 * [Project Execution Instructions](#project-execution-instructions)
 * [Introduction](#introduction)
 * [Related Work](#related-work)
 * [Results](#results)
 * [Deliverables](#deliverables)

## Project Execution Instructions

Please note: Direct Optimization (direct.py) and Bayesian Optimization (batch64.py) runs were conducted on Minnesota Supercomputing Institute (MSI) GPUs and we recommend you use GPU resources to 
reproduce the associated results. At UMN, if you do use MSI, then train.sh will allow you to schedule a job with the correct configuration. All code in PyTorch has been written to be GPU (cuda) optimized. 

1. Clone the repository
    ```bash
    git clone https://github.com/pranshupanda/accelerating-bo.git
    cd accelerating-bo
    ```
2. We highly recommend that you create and activate a virtual environment. We have provided an environment.yml file to install all necessary packages, which may be invoked with the following command. 
    ```bash
    # create conda environment
    conda env create -f environment.yml
    ```
    Activate your environment using the following command.  
    ```bash
    conda activate gpuenv
    ```

   To ensure full reproducibility of our results, the exact package versions used in the experiments are listed in environment.yml. If these versions are not compatible with your system, you may need to install or downgrade to versions that are supported by your machine. All plots will be saved in the plots folder. 
   
3. To reproduce Bayesian Optimization on LABS for Discrete Local Search, BCA, Firefly and Particle Swarm Optimization run the following command. ETA: ~24 hours
     ```bash 
     python3 batch64.py
     ```
   Optionally, if you prefer a graph to illustrate the differences please run the following command. Please note, this will fetch your results from the CSV created and produce the graph
   with values detailed in Table 2 of the report, and store them, but can only be done once the results CSV file is populated post-BO. 
    ```bash
    python3 graph.py
    ```

4. To reproduce Direct Optimization on LABS for BCA, Firefly and Particle Swarm Optimization run the following command. 
   This will likely take close to ~8.5 hours to populate the CSV 
    ```bash
    python3 direct.py
    ```
5. To reproduce kernel evaluation please run the following command, which offers the following CLI flags: `--kernel {rbf,categorical} --n INT --train-size INT --test-size INT.
    ```bash
     python3 kernel.py
    ```
6. To view an illustrative example of the working of the CHM algorithm, please run the following command.
    ```bash
    # This will generate an image within plots containing the evolution of the sequence
    python3 chm_algorithm.py
    ```
7. If you choose to run on MSI at UMN, then after activating the environment, you may run the following command after updating train.sh with your details and specifications for GPU requirements and
   update filename to reflect what you'd like to run. 
    ```bash
        sbatch train.sh
    ```

## Introduction

Protein design plays a central role in modern drug discovery and therapeutic development, yet its progress remains limited by the pace of wet-lab experimentation. Each evaluation of a candidate sequence requires synthesis, folding, and testing, creating a huge bottleneck in the experimentation and evaluation cycle. These constraints require us to consider computational methods capable of accelerating discovery by reducing the number of costly experimental evaluations. Bayesian optimization offers a powerful framework for such settings by using uncertainty-aware surrogate models and acquisition functions to guide the search for optimal designs.

Although several state-of-the-art methods for optimizing black-box and expensive-to-evaluate objective functions in continuous high-dimensional domains exist, they typically rely on gradient-based techniques to efficiently navigate the large search space. Unfortunately, gradients are undefined in discrete domains such as those found in protein design and thus renders our objective outside the scope of these methods. More importantly, one desideratum for our optimizer is its capability to make large jumps in the search space and consequently progress efficiently. We propose to leverage a new class of evolutionary algorithms capable of large jumps in the search space to make progress efficiently while being equally adept at local hill climbing compared to existing optimization methods.

Our goal for sample-efficient protein design will place the Bayesian optimization framework at the core of our work and is also one of the pillars of this course. Furthermore, we will use advancements in evolutionary algorithms in tandem with acquisition functions and their optimizers to efficiently optimize black-box, expensive-to-evaluate objectives with discrete inputs which quantify the utility of candidate proteins.

## Related Work

[1] Maximilian Balandat, Brian Karrer, Daniel R. Jiang, Samuel Daulton, Benjamin Letham, An-
drew Gordon Wilson, and Eytan Bakshy. BoTorch: A Framework for Efficient Monte-Carlo
Bayesian Optimization. In Advances in Neural Information Processing Systems 33, 2020. URL
http://arxiv.org/abs/1910.06403.

[2] Dogan Corus, Pietro S. Oliveto, and Donya Yazdani. Fast contiguous somatic hypermutations for single-
objective optimisation and multi-objective optimisation via decomposition. Proceedings of the AAAI
Conference on Artificial Intelligence, 39:26922–26930, 2025. doi: 10.1609/aaai.v39i25.34897. URL
https://doi.org/10.1609/aaai.v39i25.34897.

[3] Thomas Jansen and Christine Zarges. Analyzing different variants of immune inspired somatic con-
tiguous hypermutations. Theoretical Computer Science, 412(6):517–533, 2011. ISSN 0304-3975.
doi: https://doi.org/10.1016/j.tcs.2010.09.027. URL https://www.sciencedirect.com/science/
article/pii/S0304397510005062. Theoretical Aspects of Artificial Immune Systems.

[4] Johnny Kelsey and Jon Timmis. Immune inspired somatic contiguous hypermutation for function
optimisation. In Genetic and Evolutionary Computation — GECCO 2003, pages 207–218, Berlin,
Heidelberg, 2003. Springer Berlin Heidelberg.

[5] J. Kennedy and R. Eberhart. Particle swarm optimization. In Proceedings of ICNN’95 - International
Conference on Neural Networks, volume 4, pages 1942–1948 vol.4, 1995. doi: 10.1109/ICNN.1995.
488968.

[6] Tom Packebusch and Stephan Mertens. Low autocorrelation binary sequences. Journal of Physics A:
Mathematical and Theoretical, 49(16):165001, March 2016. ISSN 1751-8121. doi: 10.1088/1751-8113/
49/16/165001. URL http://dx.doi.org/10.1088/1751-8113/49/16/165001.

[7] Xin-She Yang. Firefly algorithms for multimodal optimization, 2010. URL https://arxiv.org/
abs/1003.1466.


## Results

## Deliverables
 * Project Progress Presentation
 * Project Report
 * Project Poster
