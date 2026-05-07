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
First, create a Conda environment using the following command.
```bash
# create conda environment
conda env create -f environment.yml
```
Then, use the following commands to install `skydiscover`.
```bash
# install skydiscover (editable mode)
conda activate protein-design-env
cd src/skydiscover
python -m pip install -e .
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
