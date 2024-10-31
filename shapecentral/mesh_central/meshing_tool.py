import logging
import os
import platform
import shutil
import subprocess
import sys

import tempfile
from os.path import join, dirname

from shapecentral.mesh_central.convert_mesh_tool import convert_to_old_format

# import treefiles as tf

# This module requires gmsh (https://gmsh.info/) and mmg (https://www.mmgtools.org/)
# mmg executables are included for windows, linux and mac for 3d and surface remeshing

if shutil.which("gmsh") is None:
    raise ImportError(
        "'gmsh' was not found!\nYou can install it with `pip install gmsh-sdk` (see https://github.com/mrkwjc/gmsh-sdk)"
    )

ssw = {}
if platform.system() == "Windows":
    ssw = {"shell": True}


def remesh(_in, _out, mmg_opt):
    """
    Wrapper to use mmg
    """

    softs = {
        "Windows": ".exe",
        "Darwin": "_osx",
        "Linux": "",
    }
    s = platform.system()
    if s not in softs:
        # log.error(f"OS not recognized: {s!r}")
        raise RuntimeError(f"OS not recognized: {s!r}")
    # soft = tf.f(__file__, "mmg", "5.6.0") / "mmgs_O3" + softs[s]
    soft = join(dirname(__file__), "mmg", "5.6.0", "mmgs_O3" + softs[s])

    # log.debug("Start remeshing")

    std_out = subprocess.check_output(
        [
            soft,
            "-in",
            _in,
            "-out",
            _out,
            *list(map(str, mmg_opt)),
        ],
        **ssw,
    )
    # log.debug(std_out)
    # log.info(f"Remeshing with mmg ok")


# @tf.timer
def triangulate(
        _input,
        output,
        mmg=True,
        mmg_opt=None,
):
    """
    Will triangulate the mesh given as `_input` and produce and
    unstructured mesh, wrote to filename `output`.

    :param str _input: Input mesh, object_loadable polydata
    :param str output: Output unstructured mesh filename
    :param bool mmg: Choose to remesh with mmg
    :param list mmg_opt: mmg options, see https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-options
    """

    if mmg_opt is None:
        mmg_opt = ["-hmax", 3, "-hgrad", 1, "-hausd", 10, "-nr"]
    convert_to_old_format(_input)
    _input = os.path.abspath(_input)
    root = os.path.splitext(output)[0]
    gmsh_out = root + "_gmsh_out" + ".mesh"
    mmg_out = root + "_mmg_out" + ".mesh"

    geo_template = f"Merge {_input!r};"
    geo_template += "\nSurface Loop(1) = {1};\n//+\nVolume(1) = {1};"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".geo") as f:
        f.write(geo_template)
        f.flush()

        subprocess.call(["gmsh", f.name, "-o", gmsh_out, "-save"])

    if not mmg:
        mmg_out = gmsh_out
    else:
        remesh(_in=gmsh_out, _out=mmg_out, mmg_opt=mmg_opt)

    subprocess.call(["gmsh", mmg_out, "-o", output, "-save"])

    mmg_sol = root + "_mmg_out" + ".sol"
    if os.path.exists(gmsh_out):
        os.remove(gmsh_out)
    if os.path.exists(mmg_out):
        os.remove(mmg_out)
    if os.path.exists(mmg_sol):
        os.remove(mmg_sol)


log = logging.getLogger(__name__)
log.disabled = True
