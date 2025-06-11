# pylint: disable=C0114
from onescience.models.protenix.config.extend_types import ListValue, RequiredValue

inference_configs = {
    "seeds": ListValue([101]),
    "dump_dir": "./output",
    "need_atom_confidence": False,
    "input_json_path": RequiredValue(str),
    "load_checkpoint_path": RequiredValue(str),
    "num_workers": 16,
    "use_msa": True,
}
