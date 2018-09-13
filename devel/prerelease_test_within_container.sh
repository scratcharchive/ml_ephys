#!/bin/bash

conda install -c conda-forge -c flatiron mountainlab
ml-config

pip install .
ln -s $PWD `ml-config package_directory`/ml_ephys

ml-list-processors

ml-run-process ephys.synthesize_random_waveforms -o waveforms_out:waveforms.mda
mda-info waveforms.mda

