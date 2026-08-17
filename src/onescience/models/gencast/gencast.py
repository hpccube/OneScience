"""项目内官方等价 GenCast JAX/Haiku 实现封装。"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import haiku as hk
import jax
import xarray

from .graphcast import checkpoint
from .graphcast import denoiser
from .graphcast import gencast
from .graphcast import nan_cleaning
from .graphcast import normalization
from .graphcast import xarray_jax
from .graphcast import xarray_tree


def build_model_config(config: dict[str, Any]) -> tuple[
    Any, denoiser.DenoiserArchitectureConfig, gencast.SamplerConfig,
    gencast.NoiseConfig, denoiser.NoiseEncoderConfig
]:
    """Build a random-weight configuration without changing GenCast semantics."""
    model_cfg = config["model"]
    sampler_cfg = config["sampler"]
    transformer = denoiser.SparseTransformerConfig(
        attention_k_hop=int(model_cfg["attention_k_hop"]),
        d_model=int(model_cfg["latent_size"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        attention_type=str(model_cfg["attention_type"]),
        mask_type=str(model_cfg.get("mask_type", "full")),
        ffw_hidden=int(model_cfg["ffw_hidden"]),
    )
    architecture = denoiser.DenoiserArchitectureConfig(
        sparse_transformer_config=transformer,
        mesh_size=int(model_cfg["mesh_size"]),
        latent_size=int(model_cfg["latent_size"]),
        hidden_layers=int(model_cfg.get("hidden_layers", 1)),
        radius_query_fraction_edge_length=float(
            model_cfg.get("radius_query_fraction_edge_length", 0.6)
        ),
    )
    sampler = gencast.SamplerConfig(**sampler_cfg)
    return (
        gencast.TASK,
        architecture,
        sampler,
        gencast.NoiseConfig(),
        denoiser.NoiseEncoderConfig(),
    )


def load_model_checkpoint(path: str | Path) -> gencast.CheckPoint:
    """Load the typed official GenCast NPZ checkpoint."""
    with Path(path).open("rb") as source:
        return checkpoint.load(source, gencast.CheckPoint)


class GenCastModel:
    """Owns official-equivalent GenCast loss and sampling Haiku transforms."""

    def __init__(
        self,
        *,
        task_config: Any,
        architecture_config: denoiser.DenoiserArchitectureConfig,
        sampler_config: gencast.SamplerConfig,
        noise_config: gencast.NoiseConfig,
        noise_encoder_config: denoiser.NoiseEncoderConfig,
        diffs_stddev_by_level: xarray.Dataset,
        mean_by_level: xarray.Dataset,
        stddev_by_level: xarray.Dataset,
        min_by_level: xarray.Dataset,
        reintroduce_nans: bool = True,
    ) -> None:
        self.task_config = task_config
        self.architecture_config = architecture_config
        self.sampler_config = sampler_config
        self.noise_config = noise_config
        self.noise_encoder_config = noise_encoder_config
        self.diffs_stddev_by_level = diffs_stddev_by_level
        self.mean_by_level = mean_by_level
        self.stddev_by_level = stddev_by_level
        self.min_by_level = min_by_level
        self.reintroduce_nans = reintroduce_nans

        def construct() -> Any:
            predictor = gencast.GenCast(
                task_config=self.task_config,
                denoiser_architecture_config=self.architecture_config,
                sampler_config=self.sampler_config,
                noise_config=self.noise_config,
                noise_encoder_config=self.noise_encoder_config,
            )
            predictor = normalization.InputsAndResiduals(
                predictor,
                diffs_stddev_by_level=self.diffs_stddev_by_level,
                mean_by_level=self.mean_by_level,
                stddev_by_level=self.stddev_by_level,
            )
            return nan_cleaning.NaNCleaner(
                predictor,
                var_to_clean="sea_surface_temperature",
                fill_value=self.min_by_level,
                reintroduce_nans=self.reintroduce_nans,
            )

        @hk.transform_with_state
        def loss_fn(inputs, targets, forcings):
            loss, diagnostics = construct().loss(inputs, targets, forcings)
            return xarray_tree.map_structure(
                lambda value: xarray_jax.unwrap_data(
                    value.mean(), require_jax=True
                ),
                (loss, diagnostics),
            )

        @hk.transform_with_state
        def forward_fn(inputs, targets_template, forcings):
            return construct()(
                inputs,
                targets_template=targets_template,
                forcings=forcings,
            )

        self.loss_fn = loss_fn
        self.forward_fn = forward_fn

    @classmethod
    def from_config_and_stats(
        cls, config: dict[str, Any], stats: dict[str, xarray.Dataset]
    ) -> "GenCastModel":
        configs = build_model_config(config)
        return cls(
            task_config=configs[0],
            architecture_config=configs[1],
            sampler_config=configs[2],
            noise_config=configs[3],
            noise_encoder_config=configs[4],
            diffs_stddev_by_level=stats["diffs_stddev_by_level"],
            mean_by_level=stats["mean_by_level"],
            stddev_by_level=stats["stddev_by_level"],
            min_by_level=stats["min_by_level"],
            reintroduce_nans=bool(config.get("data", {}).get("reintroduce_sst_nans", True)),
        )

    @classmethod
    def from_checkpoint_and_stats(
        cls,
        model_checkpoint: gencast.CheckPoint,
        stats: dict[str, xarray.Dataset],
        *,
        attention_type: str | None = None,
    ) -> "GenCastModel":
        architecture = model_checkpoint.denoiser_architecture_config
        if attention_type is not None:
            architecture = dataclasses.replace(
                architecture,
                sparse_transformer_config=dataclasses.replace(
                    architecture.sparse_transformer_config,
                    attention_type=attention_type,
                    mask_type="full",
                ),
            )
        return cls(
            task_config=model_checkpoint.task_config,
            architecture_config=architecture,
            sampler_config=model_checkpoint.sampler_config,
            noise_config=model_checkpoint.noise_config,
            noise_encoder_config=model_checkpoint.noise_encoder_config,
            diffs_stddev_by_level=stats["diffs_stddev_by_level"],
            mean_by_level=stats["mean_by_level"],
            stddev_by_level=stats["stddev_by_level"],
            min_by_level=stats["min_by_level"],
        )

    def init(self, rng, inputs, targets, forcings):
        return self.loss_fn.init(rng, inputs, targets, forcings)

    def loss(self, params, state, rng, inputs, targets, forcings):
        return self.loss_fn.apply(params, state, rng, inputs, targets, forcings)

    def predict(self, params, state, rng, inputs, targets_template, forcings):
        return self.forward_fn.apply(
            params, state, rng, inputs, targets_template, forcings
        )


def parameter_count(params: Any) -> int:
    return sum(int(value.size) for value in jax.tree_util.tree_leaves(params))
