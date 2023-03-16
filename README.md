
# r255_bias_in_datasets
This repository supports the coursework submission of R255 Advanced Topics in Machine Learning - Bias in Datasets at the University of Cambridge, 2022/23 academic year.

## Requirements: 
The following repos must be cloned to current directory:

 - https://github.com/BrendanKennedy/contextualizing-hate-speech-models-with-explanations (Kennedy et al., 2020)
 - https://github.com/paul-rottger/hatecheck-data (Rottger et al., 2021)
 - https://github.com/g8a9/ear (Attanasio et al., 2022)
	(the folder `contextualizing-hate-speech-models-with-explanations` needs to be renamed as `soc`)

## To run SOC with GHC data
follow the instructions in Kennedy et al.'s repository.

## To run EAR with 
GHC data

    ./train_model_cust_data_10_seeds.sh bert-base-uncased ./ear_bert gab25k
Vidgen et al., data

    python ear_dyna.py

## File description
 - `custom_dataset_for_ear.py` custom dataset code for fine-tuning EAR with GHC and Learning from the Worst
- `ear_dyna.py` Fine-tuning EAR with Learning from the Worst
- `ear_with_gab.py` Fine-tuning EAR with GHC
- all `.ipynb` notebooks: for out-of-distribution evaluation and visualisations
