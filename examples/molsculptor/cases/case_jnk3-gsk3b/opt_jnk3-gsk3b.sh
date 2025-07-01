#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
ONESCIENCE_PATH=$(python -c "import onescience; print(onescience.__path__[0])")
echo $ONESCIENCE_PATH
python -u $SCRIPT_DIR/../../diff_evo_opt_dual.py \
    --params_path $ONESCIENCE_PATH/flax_models/MolSculptor/checkpoints/diffusion-transformer/dit_params_opt.pkl \
    --config_path $ONESCIENCE_PATH/flax_models/MolSculptor/checkpoints/diffusion-transformer/config_opt.pkl \
    --logger_path $SCRIPT_DIR/test/Logs.txt \
    --save_path $SCRIPT_DIR/test \
    --dsdp_script_path_1 $SCRIPT_DIR/dsdp_jnk3.sh \
    --dsdp_script_path_2 $SCRIPT_DIR/dsdp_gsk3b.sh \
    --random_seed 8888 \
    --np_random_seed 8888 \
    --total_step 30 \
    --device_batch_size 128 \
    --t_min 70 \
    --t_max 80 \
    --n_replicate 8 \
    --num_latent_tokens 16 \
    --dim_latent 32 \
    --eq_steps 10 \
    --vae_config_path $ONESCIENCE_PATH/flax_models/MolSculptor/checkpoints/auto-encoder/config.pkl \
    --vae_params_path $ONESCIENCE_PATH/flax_models/MolSculptor/checkpoints/auto-encoder/ae_params_opt.pkl \
    --alphabet_path $ONESCIENCE_PATH/flax_models/MolSculptor/train/smiles_alphabet.pkl \
    --init_molecule_path $SCRIPT_DIR/init_search_molecule.pkl \
    --sub_smiles 'O=C(N1CCNCC1)c1ccccc1'
exit