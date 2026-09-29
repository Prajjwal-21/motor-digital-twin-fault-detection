% MAIN  Build the dataset from the DC motor digital twin and plot examples.
%
% Run from the matlab/ folder:
%   >> main
% or from a terminal:
%   matlab -batch "cd matlab; main"

clear; clc;

generate_dataset();   % -> data/motor_dataset.csv, data/run_metadata.csv
plot_examples();      % -> results/matlab_example_signals.png
