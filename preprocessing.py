# TO RUN: 
# blender -b --python preprocessing.py -- --light_setup 1,2,3 --camera_setup 1
import sys, os
import bpy

sys.path.append('./')

from PDG.scene.Scene import Scene
from PDG.utils.Utils import clear_scene, save_blend_file, render_mesh, add_exif_metadata, generate_dataset_folder, generate_unet_dataset, generate_dataset
from PDG.options.base_options import BaseOptions

def create_blender_file(opt, idx):
    for dir in [d for d in os.listdir(opt.input_blender_directory_path) if os.path.isdir(os.path.join(opt.input_blender_directory_path, d))]:
        subdir_path = os.path.join(opt.input_blender_directory_path, dir)
        blend_file_path = os.path.abspath(os.path.join(subdir_path, dir.lower() + ".blend"))
        try:
            clear_scene()
            with bpy.data.libraries.load(blend_file_path, link=True) as (data_from, data_to):
                data_to.objects = data_from.objects

            scene = Scene(data_to.objects[0], config[0], config[1])
            scene.render_scene()
            bpy.context.collection.objects.link(data_to.objects[0])
            bpy.context.view_layer.update()
            save_blend_file(opt.blender_output_dir, dir.lower() + "_" + str(idx+1))           
        except Exception as e:
            print(f"Error loading blend file: {e}")

if __name__ == "__main__":

    opt = BaseOptions().parse()
    light_setup = [int(num) for num in opt.light_setup.split(',')]
    camera_setup = [int(num) for num in opt.camera_setup.split(',')]
    configurations = [(x, y) for x in light_setup for y in camera_setup]
    
    # blender -b --python preprocessing.py -- --light_setup 1,2,3 --camera_setup 1 --create_dataset False --create_unet_dataset False --render False
    if(opt.create_blender_file == "True"):
        for index, config in enumerate(configurations):
            create_blender_file(opt, index)
    
    # blender -b --python preprocessing.py -- --create_blender_file False --create_dataset False --create_unet_dataset False
    if(opt.render == "True"):
        render_mesh(opt.blender_output_dir, opt.blender_dataset_dir)

    if(opt.add_exif_metadata == "True"):
        add_exif_metadata(opt.blender_dataset_dir)

    # python preprocessing.py -- --create_blender_file False --render False
    if(opt.create_dataset == "True"):
        generate_dataset(opt.blender_dataset_dir, opt.dataset_dir, generate_unet_dataset=True, generate_pix2pix_dataset=True)
