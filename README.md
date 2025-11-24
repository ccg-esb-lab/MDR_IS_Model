# Stochastic Model of Plasmid-Encoded IS-Driven Evolution

Scripts and notebooks used to produce figures and simulations related to the theoretical model from:

**Plasmids promote antimicrobial resistance through Insertion Sequence-mediated gene inactivation**  
_Jorge Sastre-Domínguez, Paloma Rodera-Fernández, Javier DelaFuente,
Susana Quesada-Gutiérrez, Marina Velencoso-Requena, Sandra Martínez-González,
Alicia Calvo-Villamañán, Coloma Costas Romero, Ayari Fuentes-Hernández,
Alfonso Santos-López, Álvaro San Millán_.

## Overview

This repository contains the scripts and notebooks for the stochastic model developed in our study, which explores the evolutionary dynamics of plasmid-bearing and plasmid-free bacterial populations carrying insertion sequences (ISs) under antibiotic selection. The model, based on the Gillespie algorithm, simulates random events of birth, death, mutation (SNPs and ISs), plasmid segregation, and conjugation in a multispecies community competing for a limiting resource.  
Simulations explore how the rate of plasmid conjugation and the structure of transmission networks influence plasmid spread, mutation accumulation, and the shift toward IS-driven evolutionary dynamics.

## Notebooks

### [MonodGillespieIS_multispecies.ipynb](MonodGillespieIS_multispecies.ipynb)  
Describes the multispecies stochastic model and shows how birth, death, conjugation, mutation, and segregational loss are implemented under resource limitation.

### [MonodGillespieIS_parametrization.ipynb](MonodGillespieIS_parametrization.ipynb)  
Uses experimental monoculture data to calibrate strain-specific parameters, including Monod growth curves and antibiotic susceptibility from dose–response assays.

### [MonodGillespieIS_conjugation.ipynb](MonodGillespieIS_conjugation.ipynb)  
Simulates plasmid dynamics in a fully connected transfer network while varying the conjugation rate to identify thresholds for plasmid stability and shifts in mutational spectra.

### [MonodGillespieIS_networks.ipynb](MonodGillespieIS_networks.ipynb)  
Explores how reducing connectivity in the plasmid transfer network alters plasmid prevalence and dynamics of SNP- and IS-driven evolution.


## License

[MIT](https://choosealicense.com/licenses/mit/)

This project is licensed under the MIT License — see the [license.txt](license.txt) file for details.
