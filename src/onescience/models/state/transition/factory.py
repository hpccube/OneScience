from pathlib import Path

import torch


def get_lightning_module(model_type: str, data_config: dict, model_config: dict, training_config: dict, var_dims: dict):
    """Create model instance based on config."""
    module_config = {**model_config, **training_config}
    module_config["embed_key"] = data_config["embed_key"]
    module_config["output_space"] = data_config["output_space"]
    module_config["gene_names"] = var_dims["gene_names"]
    module_config["batch_size"] = training_config["batch_size"]
    module_config["control_pert"] = data_config.get("control_pert", "non-targeting")

    if data_config["output_space"] == "gene":
        gene_dim = var_dims["hvg_dim"]
    else:
        gene_dim = var_dims["gene_dim"]

    if model_type.lower() == "embedsum":
        from onescience.modules.state.transition.embed_sum import EmbedSumPerturbationModel

        return EmbedSumPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "old_neuralot":
        from onescience.modules.state.transition.old_neural_ot import OldNeuralOTPerturbationModel

        return OldNeuralOTPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() in {"neuralot", "pertsets", "state"}:
        from onescience.modules.state.transition.state_transition import StateTransitionPerturbationModel

        return StateTransitionPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            basal_mapping_strategy=data_config["basal_mapping_strategy"],
            **module_config,
        )
    elif model_type.lower() in {"globalsimplesum", "perturb_mean"}:
        from onescience.modules.state.transition.perturb_mean import PerturbMeanPerturbationModel

        return PerturbMeanPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() in {"celltypemean", "context_mean"}:
        from onescience.modules.state.transition.context_mean import ContextMeanPerturbationModel

        return ContextMeanPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "decoder_only":
        from onescience.modules.state.transition.decoder_only import DecoderOnlyPerturbationModel

        return DecoderOnlyPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "pseudobulk":
        from onescience.modules.state.transition.pseudobulk import PseudobulkPerturbationModel

        return PseudobulkPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "cpa":
        from onescience.modules.state.transition.cpa import CPAPerturbationModel

        return CPAPerturbationModel(
            input_dim=var_dims["input_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            gene_dim=gene_dim,
            **module_config,
        )
    elif model_type.lower() == "scvi":
        from onescience.modules.state.transition.scvi import SCVIPerturbationModel

        return SCVIPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() in {"scgpt-chemical", "scgpt-genetic"}:
        from onescience.modules.state.transition.scgpt import scGPTForPerturbation

        pretrained_path = module_config["pretrained_path"]
        assert pretrained_path is not None, "pretrained_path must be provided for scGPT"

        model_dir = Path(pretrained_path)
        model_file = model_dir / "best_model.pt"

        model = scGPTForPerturbation(
            ntoken=module_config["ntoken"],
            n_drug_tokens=module_config["n_perts"],
            d_model=module_config["d_model"],
            nhead=module_config["nhead"],
            d_hid=module_config["d_hid"],
            nlayers=module_config["nlayers"],
            nlayers_cls=module_config["n_layers_cls"],
            n_cls=1,
            dropout=module_config["dropout"],
            pad_token_id=module_config["pad_token_id"],
            pad_value=module_config["pad_value"],
            pert_pad_id=module_config["pert_pad_id"],
            do_mvc=module_config["do_MVC"],
            cell_emb_style=module_config["cell_emb_style"],
            mvc_decoder_style=module_config["mvc_decoder_style"],
            use_fast_transformer=module_config["use_fast_transformer"],
            lr=module_config["lr"],
            step_size_lr=module_config["step_size_lr"],
            include_zero_gene=module_config["include_zero_gene"],
            embed_key=module_config["embed_key"],
            perturbation_type=module_config["perturbation_type"],
        )

        load_param_prefixes = module_config["load_param_prefixes"]

        if load_param_prefixes is not None:
            model_dict = model.model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                key: value
                for key, value in pretrained_dict.items()
                if any(key.startswith(prefix) for prefix in load_param_prefixes)
            }
            for key, value in pretrained_dict.items():
                print(f"Loading params {key} with shape {value.shape}")

            model_dict.update(pretrained_dict)
            model.model.load_state_dict(model_dict)
        else:
            try:
                model.model.load_state_dict(torch.load(model_file))
                print(f"Loading all model params from {model_file}")
            except Exception:
                model_dict = model.model.state_dict()
                pretrained_dict = torch.load(model_file)
                pretrained_dict = {
                    key: value
                    for key, value in pretrained_dict.items()
                    if key in model_dict and value.shape == model_dict[key].shape
                }
                for key, value in pretrained_dict.items():
                    print(f"Loading params {key} with shape {value.shape}")

                model_dict.update(pretrained_dict)
                model.model.load_state_dict(model_dict)

        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")
