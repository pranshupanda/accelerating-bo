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

   To ensure full reproducibility of our results, the exact package versions used in the experiments are listed in environment.yml. If these versions are not compatible with your system, you may need to install or downgrade to versions that are supported by your machine.
   
3. To reproduce Bayesian Optimization on LABS for Discrete Local Search, BCA, Firefly and Particle Swarm Optimization run the following command. 
     ```bash
     # This will likely take close to ~16 hours to populate the CSV and print a version of Table 2 from the report to your terminal window after completion.
     # Note: The formatting will be slightly different with labels like BCA-10 implying 10 dimensional LABS problem with BCA optimizer instead of the BEST<dim> format in the paper. 
     python3 batch64.py
     ```
   Optionally, if you prefer a graph to illustrate the differences please run the following command. 
    ```bash
    # This will fetch your results from the CSV created and produce the graph with values detailed in Table 2 of the report, and store them in folder plots
    # This can only be done once the results CSV file is populated post-BO. 
    python3 graph.py
    ```

4. To reproduce Direct Optimization on LABS for BCA, Firefly and Particle Swarm Optimization run the following command. 
    ```bash
    # This will likely take close to ~8.5 hours to populate the CSV 
    python3 direct.py
    ```
5. To view an illustrative example of the working of the CHM algorithm, please run the following command.
    ```bash
    # This will generate an image within plots containing the evolution of the sequence
    python3 chm_algorithm.py
    ```
6. If you choose to run on MSI at UMN, then after activating the environment, you may run the following command after updating train.sh with your details and specifications for GPU requirements. 
    ```bash
        sbatch train.sh
    ```

## Introduction

Protein design plays a central role in modern drug discovery and therapeutic development, yet its progress remains limited by the pace of wet-lab experimentation. Each evaluation of a candidate sequence requires synthesis, folding, and testing, creating a huge bottleneck in the experimentation and evaluation cycle. These constraints require us to consider computational methods capable of accelerating discovery by reducing the number of costly experimental evaluations. Bayesian optimization offers a powerful framework for such settings by using uncertainty-aware surrogate models and acquisition functions to guide the search for optimal designs.

Although several state-of-the-art methods for optimizing black-box and expensive-to-evaluate objective functions in continuous high-dimensional domains exist, they typically rely on gradient-based techniques to efficiently navigate the large search space. Unfortunately, gradients are undefined in discrete domains such as those found in protein design and thus renders our objective outside the scope of these methods. More importantly, one desideratum for our optimizer is its capability to make large jumps in the search space and consequently progress efficiently. We propose to leverage a new class of evolutionary algorithms capable of large jumps in the search space to make progress efficiently while being equally adept at local hill climbing compared to existing optimization methods.

Our goal for sample-efficient protein design will place the Bayesian optimization framework at the core of our work and is also one of the pillars of this course. Furthermore, we will use advancements in evolutionary algorithms in tandem with acquisition functions and their optimizers to efficiently optimize black-box, expensive-to-evaluate objectives with discrete inputs which quantify the utility of candidate proteins.

## Related Work

## Results

## Deliverables
 * Project Progress Presentation
 * Project Report
 * Project Poster
