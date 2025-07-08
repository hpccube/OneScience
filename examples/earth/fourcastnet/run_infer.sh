echo $ONESCIENCE_MODELS_DIR

python inference.py --vis --yaml_config=./conf/AFNO.yaml --config=afno_backbone --run_num=check_exp --weight=${ONESCIENCE_MODELS_DIR}/FourCastNet/best_ckpt.tar