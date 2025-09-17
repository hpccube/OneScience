from dataclasses import dataclass


@dataclass
class DatapipeMetaData:
    """Data class for storing essential meta data needed for all onescience datapipes"""

    # Datapipe info
    name: str = "OnescienceDatapipe"
    # Optimizations
    auto_device: bool = False
    cuda_graphs: bool = False
    # Parallel
    ddp_sharding: bool = False
