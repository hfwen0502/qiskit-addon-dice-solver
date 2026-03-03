# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test SBD solver with N2 molecule using subprocess-based wrapper."""

import os

import numpy as np
from pyscf import ao2mo, tools

from qiskit_addon_dice_solver.sbd_solver import solve_sci_batch
from qiskit_addon_sqd.counts import generate_bit_array_uniform
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian

# Specify molecule properties
num_orbitals = 16
num_elec_a = num_elec_b = 5
spin_sq = 0

# Read in molecule from disk
active_space_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "molecules", "n2_fci.txt"
)
mf_as = tools.fcidump.to_scf(active_space_path)
hcore = mf_as.get_hcore()
eri = ao2mo.restore(1, mf_as._eri, num_orbitals)
nuclear_repulsion_energy = mf_as.mol.energy_nuc()

# Create a seed to control randomness throughout this workflow
rand_seed = np.random.default_rng(42)

# Generate random samples
bit_array = generate_bit_array_uniform(10_000, num_orbitals * 2, rand_seed=rand_seed)


# Run SQD with SBD solver
result_history = []


def callback(results: list[SCIResult]):
    result_history.append(results)
    iteration = len(result_history)
    print(f"Iteration {iteration}")
    for i, result in enumerate(results):
        print(f"\tSubsample {i}")
        print(f"\t\tEnergy: {result.energy + nuclear_repulsion_energy}")
        print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")


def solve_sci_batch_with_sbd(ci_strings, one_body_tensor, two_body_tensor, norb, nelec):
    """Wrapper that adds SBD-specific options to solve_sci_batch."""
    import os
    sbd_path = os.environ.get("SBD_EXECUTABLE_PATH")
    return solve_sci_batch(
        ci_strings,
        one_body_tensor,
        two_body_tensor,
        norb,
        nelec,
        sbd_path=sbd_path,  # Use environment variable or PATH
        mpirun_options=["-np", "1"],  # MPI options
        sbd_config={
            "method": 0,  # Davidson
            "max_it": 100,
            "max_nb": 50,
            "tolerance": 1e-8,
            "carryover_type": 1,  # Marginal probabilities
            "carryover_ratio": 0.1,
            "carryover_threshold": 1e-4,
            "bit_length": 64,
            "do_rdm": 1,  # Full RDM (changed from 0)
            "task_comm_size": 1,
            "adet_comm_size": 1,
            "bdet_comm_size": 1,
        },
        temp_dir="/tmp/sbd",
        clean_temp_dir=False,  # Keep files for debugging
    )


result = diagonalize_fermionic_hamiltonian(
    hcore,
    eri,
    bit_array,
    samples_per_batch=300,
    norb=num_orbitals,
    nelec=(num_elec_a, num_elec_b),
    num_batches=5,
    max_iterations=2,  # Reduced to 2 iterations for testing
    sci_solver=solve_sci_batch_with_sbd,
    symmetrize_spin=True,
    callback=callback,
    seed=rand_seed,
)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Exact energy:     -109.10288938 Hartree")
print(f"Estimated energy: {result.energy + nuclear_repulsion_energy:.8f} Hartree")
print(f"Error:            {abs(result.energy + nuclear_repulsion_energy + 109.10288938):.8e} Hartree")
print("="*70)

# Made with Bob
