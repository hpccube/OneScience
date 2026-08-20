"""Training, embedding, and perturbation workflows for Geneformer.

Public workflow classes are imported lazily so lightweight data modules do not
initialize optional training and quantization dependencies.
"""

from importlib import import_module


_EXPORTS = {
    "Classifier": ("classifier", "Classifier"),
    "EmbExtractor": ("emb_extractor", "EmbExtractor"),
    "GeneformerPretrainer": ("pretrainer", "GeneformerPretrainer"),
    "InSilicoPerturber": ("in_silico_perturber", "InSilicoPerturber"),
    "InSilicoPerturberStats": (
        "in_silico_perturber_stats",
        "InSilicoPerturberStats",
    ),
    "MTLClassifier": ("mtl_classifier", "MTLClassifier"),
    "get_embs": ("emb_extractor", "get_embs"),
}


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)
