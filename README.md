# Project README

## Abstract

This repository contains the implementation, training, and evaluation framework for multiple algorithms organized under a unified project structure. The `Main/` directory includes the primary execution script for evaluation as well as the subdirectories of all supported algorithms. Running `main.py` from `Main/` generates performance outputs such as the BER curve and CCDF, while each algorithm can be trained independently through its own `train.py` script located in the corresponding subdirectory. To simplify installation and ensure reproducibility, the project also provides a `requirements.txt` file and a `Dockerfile`.

## Repository Structure

```text
.
├── Main/
│   ├── main.py
│   ├── algo_1/
│   │   └── train.py
│   ├── algo_2/
│   │   └── train.py
│   └── ...
├── Dockerfile
└── requirements.txt

All algorithm directories are located inside Main/.

How to Run

First, move into the Main/ directory:

cd /Main
Generate BER Curve and CCDF

To generate the BER curve and CCDF, run:

python main.py
Train Each Algorithm

Each algorithm has its own subdirectory inside Main/.

To train a specific algorithm, navigate to its directory and run:

python train.py

For example:

cd /Main/<algorithm_directory>
python train.py

Repeat this for each algorithm directory respectively.

Environment Setup
Using requirements.txt

Install the required Python packages with:

pip install -r requirements.txt
Using Docker

A Dockerfile is also provided for building and running the project in a Docker environment.