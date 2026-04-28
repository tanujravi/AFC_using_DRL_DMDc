from smartsim_util import smartSimUtil
import sys
import os
import time
import glob
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch as pt
from yaml import safe_load
from os.path import join
from smartsim import Experiment
from smartsim.settings import RunSettings
from smartsim.status import SmartSimStatus
from smartredis import Client
import shutil
import re

class smartSimExecution:

    @staticmethod
    def update_openfoam_control_dicts(t_start: float, t_end: float):
        """Write time and omega into OpenFOAM dictionaries."""
        control_dict = os.path.join("system", "controlDict")
        os.system(f"foamDictionary {control_dict} -entry startTime -set {t_start}")
        os.system(f"foamDictionary {control_dict} -entry endTime -set {t_end}")
        """
        os.system(
            f"foamDictionary 0.org/U "
            f"-entry \"boundaryField.cylinder.omega\" -set {omega_act}"
        )
        """
    @staticmethod
    def reconstruct_fields_with_svdtofoam(
        exp: Experiment,
        svd_rank: int,
        fo_name: str,
        mpi_ranks: int,
        fields: List[str],
        path: str,
    ):
        """Call svdToFoam for each field to reconstruct into OpenFOAM."""
        for field in fields:
            settings = exp.create_run_settings(
                exe="svdToFoam",
                exe_args=(
                    f"-fieldName {field} -svdRank {svd_rank} "
                    f"-FOName {fo_name} -parallel"
                ),
                run_command="mpirun",
                run_args={"np": f"{mpi_ranks}"},
            )
            model = exp.create_model(
                name=f"svdToFoam_r{svd_rank}_field_name_{field}",
                run_settings=settings,
                path=path,
            )
            exp.start(model, summary=False, block=False)
            smartSimUtil.wait_for_completion(exp, model)

    @staticmethod
    def run_forces_postprocess(exp: Experiment, mpi_ranks: int, path: str):
        """Run pimpleFoam in postProcess mode to compute forces on reconstructed fields."""
        settings = exp.create_run_settings(
            exe="pimpleFoam",
            exe_args="-postProcess -dict system/FO_force -parallel",
            run_command="mpirun",
            run_args={"np": f"{mpi_ranks}"},
        )
        model = exp.create_model(
            name=f"forces_reconstructed_fields",
            run_settings=settings,
            path=path,
        )
        exp.start(model, summary=False, block=False)
        smartSimUtil.wait_for_completion(exp, model)

