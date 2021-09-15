#!/usr/bin/env python3
# call via `blender --background --factory-startup --python thisfile.py -- --option --option <files>
#

import argparse
from glob import glob
import logging as lg
import os
import re
import subprocess
import sys

from typing import Dict, List, Tuple


# Sadly, execBlender can't be defined later on, so it ends up having to
# sit right in the middle of our imports!
#
# ? Should we redirect stdout/stderr before execing blender?
def execBlender(reason: str):
    blender_bin = "blender"

    print("Not running under blender (%s)" % (reason))
    print("Re-execing myself under blender (blender must exist in path)...")

    mypath = os.path.realpath(__file__)

    # Windows needs munging to handle spaces in the path ... sigh
    if os.name == 'nt':
        mypath = f"\"{mypath}\""

    blender_args = [
        blender_bin,
        "--background",
        "--factory-startup",
        "--python",
        mypath,
        "--",
    ] + sys.argv[1:]

    print("executing: %s" % (" ".join((blender_args))))

    # For some reason, this just doesn't work under Windows if there's a
    # space in the path. Can't manage to feed it anything that will actually
    # work, despite the same command line as I can run by hand.
    try:
        os.execvp(blender_bin, blender_args)
    except OSError as e:
        print("Couldn't exec blender: %s" % (e))
        sys.exit(1)


# Check if we're running under Blender ... and if not, fix that.
# We both have to check to make sure we can import bpy, *and* check
# to make sure there's something meaningful inside that module (like
# an actual context) because there exist 'stub' bpy modules for
# developing outside of blender, that will still import just fine...)
try:
    import bpy
except ImportError:
    execBlender("no bpy available")

# It imported ok, so now check to see if we have a context object
if bpy.context is None:
    execBlender("no context available")


#
# actual non-boootstrap code
#

#
# basic blender prep
def init_blender() -> None:
    # bpy.context.preferences.view.show_splash = False
    # bpy.context.preferences.filepaths.use_save_preview_images = False
    # bpy.context.space_data.shading.type = 'MATERIAL')
    # bpy.ops.preferences.addon_enable(module="WoWbjectImporter")
    bpy.ops.preferences.addon_enable(module="render_auto_tile_size")

    # FIXME: make addon loading configurable
    # bpy.ops.preferences.addon_enable(module="GearEngine")
    bpy.context.scene.ats_settings.use_optimal = False


# The juggling required to get this to actually get this to enable
# correctly is... precise. Courtesy https://blender.stackexchange.com/a/187968
# FIXME: How can this support Optix?
def enable_gpus(device_type='CUDA'):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cuda_devices, opencl_devices = cycles_preferences.get_devices()

    if device_type == "CUDA":
        devices = cuda_devices
    elif device_type == "OPENCL":
        devices = opencl_devices
    else:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []

    for device in devices:
        if device.type == "CPU":
            device.use = False
        else:
            device.use = True
            activated_gpus.append(device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus


def set_cycles_renderer(scene: bpy.types.Scene):
    scene.render.engine = 'CYCLES'
    enable_gpus('CUDA')

    scene.cycles.feature_set = 'EXPERIMENTAL'
    scene.cycles.tile_order = 'CENTER'



def render(args, outfile):
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = 'CUDA'

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.feature_set = 'SUPPORTED'
    scene.render.resolution_percentage = args.scale
    scene.render.filepath = os.path.realpath(outfile)
    scene.cycles.use_denoising = args.denoise
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    # scene.cycles.denoiser = 'OPTIX'
    scene.cycles.samples = args.samples
    scene.cycles.use_progressive_refine = False

    scene.cycles.tile_order = 'CENTER'

    # Really, these scenes are simple enough that GPU vs. CPU, and one tile
    # size vs another probably doesn't matter. But lets set 'em anyhow.
    if args.cpu:
        scene.cycles.device = 'CPU'
        scene.render.tile_x = 32
        scene.render.tile_y = 32
    else:
        scene.cycles.device = 'GPU'
        scene.render.tile_x = 256
        scene.render.tile_y = 256

    # print(f"tile x: {scene.render.tile_x}   y: {scene.render.tile_y}")

    m = output_re.search(outfile)
    basename = m.group(1)
    filetype = str(m.group(2))

    if filetype == "png":
        scene.render.image_settings.file_format = "PNG"
    elif filetype == "jpg" or filetype == "jpeg":
        scene.render.image_settings.file_format = "JPEG"
    else:
        print(f"ERROR: unknown file extension for {os.path.basename(outfile)}")
        return

    print(
        f"INFO: output to render with size={scene.render.resolution_percentage}%,"
        f" denoising={args.denoise}, samples={args.samples}"
    )

    # FIXME: Should this be in main or somewhere else that's not here?
    if args.keep_blend:
        bpy.ops.wm.save_mainfile(filepath=basename + ".blend")

    if args.no_render:
        print("WARNING: Not rendering preview image due to --no-render")
    else:
        # FIXME: benchmark/validate the automatic tile size bit
        bpy.ops.preferences.addon_enable(module="render_auto_tile_size")
        bpy.ops.render.render(
            animation=False, write_still=True, use_viewport=False)



# arg handling
class NegateAction(argparse.Action):
    def __call__(self, parser, ns, values, option):
        setattr(ns, self.dest, option[2:4] != 'no')


def valid_file(f: str) -> str:
    if not os.path.isfile(f) or not os.access(f, os.R_OK):
        raise argparse.ArgumentError(f"'{f}' is not a readable file")

    # else
    return f

    # if f.endswith(".blend"):
    #     return readable_file(f)

    # # Not a blend file, so see if its in our scene directory
    # mydir = os.path.realpath(os.path.dirname(__file__))
    # scenepath = os.path.join(mydir, "scenes", f"texrender_scene_{f}.blend")
    # if not os.path.isfile(scenepath):
    #     raise argparse.ArgumentError(f"scene name '{f}' is not valid")

    # return scenepath


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        # FIXME: We need to specify this, because we have no meaningful
        # argv[0], but we probably shouldn't just hardcode it
        prog='lrender.py',
        description='Render an animation from one or more view layers',
    )

    parser.add_argument(
        "--debug",
        action='store_const',
        const=True,
        default=False,
        # help="Read objects and prepare them for decimation",
    )

    # parser.add_argument(
    #     "-o",
    #     "--out",
    #     type=str,
    #     # default="preview.png",
    #     default=None,

    #     help="file to output material preview to",
    # )

    parser.add_argument(
        "-l",
        "--layer",
        action='append',
        default=[],
        help="specify layer to render"
    )

    parser.add_argument(
        "--scene",
        action='store',
        default="Scene",
        help="specify scene to render"
    )

    parser.add_argument(
        "--compute_type",
        type=str.lower,
        action='store',
        default="CUDA",
        choices=[ "CUDA" ],

        help="GPU compute type to use",
    )

    parser.add_argument(
        "--renderer",
        type=str.lower,
        action='store',
        default="eevee",
        choices=[ "eevee", "cycles" ],

        help="renderer to use",
    )

    parser.add_argument(
        "--cycles-samples",
        type=int,
        action='store',
        default=0,

        help="samples to use (0 for blendfile amount)"
    )

    # parser.add_argument(
    #     "-s",
    #     "--scene",
    #     help="blender scene file to use for rendering",
    #     type=scene_file,
    #     default="sphere",
    # )

    # parser.add_argument(
    #     "-sc",
    #     "--scale",
    #     help="scale scene by x%%",
    #     type=int,
    #     default=100,
    # )

    # parser.add_argument(
    #     "--cpu",
    #     help="render with CPU instead of GPU",
    #     default=False,
    #     action='store_true',
    # )

    # parser.add_argument(
    #     "-sa",
    #     "--samples",
    #     default=16,
    #     type=int,
    #     help="number of samples to use when rendering",
    # )

    # parser.add_argument(
    #     "--denoise",
    #     "--no-denoise",
    #     dest="denoise",
    #     default=True,
    #     action=NegateAction,
    #     nargs=0,
    # )

    # parser.add_argument(
    #     "-n",
    #     "--no-render",
    #     default=False,
    #     action='store_true',
    #     help="prep blend, but don't rendder (implies --keep-blend)",
    # )

    parser.add_argument(
        "file",
        help="blender file to render",
        type=valid_file,
        # nargs=1,
    )

    parsed_args = parser.parse_args(argv)

    return parsed_args


def main(argv: List[str]) -> int:
    # print("lrender version: %s" % (git_version()))

    # When we get called from blender, the entire blender command line is
    # passed to us as argv. Arguments for us specifically are separated
    # with a double dash, which makes blender stop processing arguments.
    # If there's not a double dash in argv, it means we can't possibly
    # have any arguments, in which case, we should blow up.
    if (("--" in argv) == False):
        print("Usage: blender --background --python thisfile.py -- <args>")
        return 1

    # chop argv down to just our arguments
    args_start = argv.index("--") + 1
    argv = argv[args_start:]

    args = parse_arguments(argv)

    loglevel = "DEBUG" if args.debug else "INFO"
    LOG_FORMAT = "[%(filename)s:%(lineno)s:%(funcName)s] (%(name)s) %(message)s"
    lg.basicConfig(level=loglevel, format=LOG_FORMAT)

    log = lg.getLogger()

    init_blender()

    if not os.path.isfile(args.file):
        log.error(f"input file '{args.file}'' does not exist")
        return 1

    bpy.ops.wm.open_mainfile(filepath=args.file, load_ui=False, use_scripts=True)

    if args.scene not in bpy.data.scenes:
        log.error(f"requested scene '{args.scene}' does not exist in blendfile")
        return 1

    bpy.context.window.scene = bpy.data.scenes[args.scene]

    outfile = "output.png"
    return 0



    # # directory mode
    # # FIXME: Can we split this out somehow?
    # if args.directory:
    #     # RIght now, only accept a single positional argument
    #     # FIXME: Accept more than one directory!
    #     dirmode_prep(args)

    # # Theoretically you could specify --no-render and then not specify
    # # --keep-blend, but at that point there's not really a point, so
    # # go ahead and assume --keep-blend
    # if args.no_render:
    #     args.keep_blend = True

    # if not scene_prep(args, args.files):
    #     print("ERROR: Scene prep failed")
    #     return 1

    # if args.analyze:
    #     return 0

    # render(args, args.out)

    for layer in bpy.context.scene.view_layers:
        l.use = (l.name in args.layer)

    return 0


if __name__ == "__main__":
    ret = main(sys.argv)

    if ret != 0:
        # FIXME: How *do* we want to handle failures?
        print(f"WARNING: lrender exiting with return code {ret}")

    # How do we make blender exit with an error code? Can we?
    bpy.ops.wm.quit_blender()

    # Should never be reached
    sys.exit(0)
