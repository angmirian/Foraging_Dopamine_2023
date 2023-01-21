# Foraging_Dopamine_2023
Author: Angela Ianni
Date uploaded: January 21, 2023
Description: These are scripts used for analysis for the paper submitted to Nature Communications in January 2023 titled "Patterns of Variability in Dopamine Synthesis Capacity and Receptor Availability Measured in-vivo with PET Imaging in Healthy Humans are Associated with Trading Reward against Time Costs during Foraging"

Scripts included:
1. commands_for_paper_2022.m - this script runs the behavioral and ROI analyses including generating plots, running multiple linear regressions, and PCA analysis
2. calc_optimal_exit.m - this script runs the MVT simulations to calculate the optimal leaving thresholds for this foraging task given the specific parameters used for travel time and decay rate
3. normaliseNaN.m - slight variation of the normalize.m built-in matlab function that can handle NaN values (since some subjs have missing data)
