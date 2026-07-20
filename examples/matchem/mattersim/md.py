import argparse

from ase.build import bulk

from onescience.utils.mattersim import molecular_dynamics


parser = argparse.ArgumentParser(description="MatterSim molecular dynamics")
parser.add_argument(
    "--checkpoint",
    default=None,
    help="Path to MatterSim checkpoint. Defaults to $ONESCIENCE_MODELS_DIR/mattersim/mattersim-v1.0.0-1M.pth",
)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
args = parser.parse_args()

atoms = bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))
dynamics = molecular_dynamics(
    atoms,
    checkpoint=args.checkpoint,
    ensemble="nvt_berendsen",
    temperature=300,
    timestep=1.0,
    logfile="-",
    device=args.device,
)
dynamics.run(10)
