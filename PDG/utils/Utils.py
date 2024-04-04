import bpy
import math
import numpy as np
import os
import shutil
import random
from subprocess import run


config = {'camera': 'Canon', 
          'model': 'CanonR6_MARKII',
          'res_x': 2000,
          'res_y': 2000,
          'camera_lens': 35,
          'sensor_width': 35.9,
          'sensor_height': 24,
          }

l_c_setup = {'light': {
'light_setup_1' : 1,
'light_setup_2' : 2,
'light_setup_3' : 3,
}, 'camera': {
'camera_setup_1' : 1,
'camera_setup_2' : 2,
}}

def get_collection(name: str = "pdg_collection", type: str = "subcollection") -> bpy.types.Collection:

    collection = bpy.data.collections.get(name)
    if not collection:
        collection = bpy.data.collections.new(name)
        if type == "collection":
            bpy.context.scene.collection.children.link(collection)
    return collection

def activate_collection(name: str = "pdg_collection"):
    def find_collection(root_collection, target_name):
        # Depth-first search to find the collection by name
        for collection in root_collection.children:
            if collection.name == target_name:
                return collection
            else:
                subcollection = find_collection(collection, target_name)
                if subcollection:
                    return subcollection
        return None
    
    # Search for the collection recursively
    collection = find_collection(bpy.context.view_layer.layer_collection, name)
    bpy.context.view_layer.active_layer_collection = collection

def _fibonacci_pt(R: float, i:int, n:int):
    Phi = math.sqrt(5) * .5 + .5
    phi = 2.0*math.pi * (i / Phi - math.floor(i/Phi))
    cosTheta = 1.0 - (2 * i + 1.0)/n
    sinTheta = 1 - cosTheta * cosTheta
    sinTheta = math.sqrt(min(1, max(0, sinTheta)))
    x = math.cos(phi) * sinTheta * R
    y = math.sin(phi) * sinTheta * R
    z = cosTheta * R
    return (x, y, z)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate(point, angle_degrees, axis=(0,1,0)):
    theta_degrees = angle_degrees
    theta_radians = math.radians(theta_degrees)

    rotated_point =  np.dot(rotation_matrix(axis, theta_radians), point)
    
    return rotated_point

def clear_scene():
    # Clear all objects from the scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.select_by_type(type='EMPTY')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.select_by_type(type='CAMERA')
    bpy.ops.object.delete()

    # Reset the scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

def setup_nodetree_composite(file_name):

    directory_path = "./PDG/output/" + file_name + "/"
    file_path_output = os.path.abspath(directory_path)

    # Set the render engine to Eevee
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    # Set the render resolution
    bpy.context.scene.render.resolution_x = 2000  
    bpy.context.scene.render.resolution_y = 2000

    bpy.context.scene.render.filepath = file_path_output + "\COMBINED/"

    bpy.context.scene.render.use_overwrite = False

    #Enable RENDER -> SAMPLING -> RENDER
    bpy.context.scene.eevee.taa_render_samples = 128 # default 64
    #Enable RENDER -> SAMPLING -> VIEWPORT
    bpy.context.scene.eevee.taa_samples = 64 # default 16   

    # Enable Ambient Occlusion
    bpy.context.scene.eevee.use_gtao = True
    bpy.context.scene.eevee.use_gtao_bounce = True

    # Set Ambient Occlusion settings
    bpy.context.scene.eevee.gtao_distance = 0.2 
    bpy.context.scene.eevee.gtao_factor = 1.0
    
    # Set size of point and area light shadow maps
    bpy.context.scene.eevee.shadow_cube_size = '2048'
    # Set size of sun light shadow maps
    bpy.context.scene.eevee.shadow_cascade_size = '2048'

    # Use 32-bit shadows
    bpy.context.scene.eevee.use_shadow_high_bitdepth = True
    
    # switch on nodes and get reference
    bpy.context.scene.use_nodes = True

    #Enable PASSES -> DATA -> Combined
    bpy.context.view_layer.use_pass_combined = True

    #Enable PASSES -> Light -> Diffuse -> Light
    bpy.context.view_layer.use_pass_diffuse_direct = True

    #Enable PASSES -> Light -> Diffuse -> Color
    bpy.context.view_layer.use_pass_diffuse_color = True

    #Enable PASSES -> Light -> Glossary (specular) -> Light
    bpy.context.view_layer.use_pass_glossy_direct = True

    #Enable PASSES -> Light -> Glossary (specular) -> Color
    bpy.context.view_layer.use_pass_glossy_color = True

    # Create a new Compositor node tree
    compositor_tree = bpy.context.scene.node_tree

    render_layer_node = None

    # Clear existing nodes
    for node in compositor_tree.nodes:
        if node.type == 'R_LAYERS':
            render_layer_node = node
            render_layer_node.location = (0, 0)
        else:
            node.location = (350, 500)

    # Create mix color node 1
    mix_color_node_1 = compositor_tree.nodes.new(type='CompositorNodeMixRGB')
    mix_color_node_1.location = (500, 300)  
    mix_color_node_1.blend_type = 'MULTIPLY'    

    # Create mix color node 2
    mix_color_node_2 = compositor_tree.nodes.new(type='CompositorNodeMixRGB')
    mix_color_node_2.location = (500, -300)  
    mix_color_node_2.blend_type = 'MULTIPLY'

    # Create output file node 1
    output_file_node_1 = compositor_tree.nodes.new(type='CompositorNodeOutputFile')
    output_file_node_1.location = (1000, 300)
    output_file_node_1.base_path = file_path_output + "\DIFFUSE"

    # Create output file node 2
    output_file_node_2 = compositor_tree.nodes.new(type='CompositorNodeOutputFile')
    output_file_node_2.location = (1000, -300)
    output_file_node_2.base_path = file_path_output + "\SPECULAR"

    compositor_tree.links.new(render_layer_node.outputs['DiffDir'], mix_color_node_1.inputs[1])
    compositor_tree.links.new(render_layer_node.outputs['DiffCol'], mix_color_node_1.inputs[2])
    compositor_tree.links.new(render_layer_node.outputs['GlossDir'], mix_color_node_2.inputs[1])
    compositor_tree.links.new(render_layer_node.outputs['GlossCol'], mix_color_node_2.inputs[2])

    compositor_tree.links.new(mix_color_node_1.outputs['Image'], output_file_node_1.inputs[0])
    compositor_tree.links.new(mix_color_node_2.outputs['Image'], output_file_node_2.inputs[0])

def save_blend_file(output_dir, file_name):
    setup_nodetree_composite(file_name)

    # Set the file path where you want to save the .blend file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = file_name + '.blend' 
    blend_file_path_output = os.path.abspath(os.path.join(output_dir, file_name))
    
    # Save the current blend file
    bpy.ops.wm.save_as_mainfile(filepath=blend_file_path_output)
    #print(f"File {file_name} salvato!")

def render_mesh(directory_path = "./PDG/blender_model_output/", output_dir = "./PDG/blender_data/"):
    blend_files = [file for file in os.listdir(directory_path) if file.endswith('.blend')]
    for blend_file in blend_files:
        # Open the .blend file
        blend_file_path = os.path.abspath(os.path.join(directory_path, blend_file))
        bpy.ops.wm.open_mainfile(filepath=blend_file_path)
        
        # Get the current scene
        scene = bpy.context.scene

        # Get the compositing node tree
        node_tree = scene.node_tree

        # Get all cameras in the scene
        cameras = [obj for obj in bpy.context.view_layer.objects if obj.type == 'CAMERA']

        output_path = output_dir + blend_file.split(".")[0] + '/'
        combined_file_path_output = os.path.abspath(os.path.join(output_path, 'COMBINED/'))
        diffused_file_path_output = os.path.abspath(os.path.join(output_path, 'DIFFUSE/'))
        specular_file_path_output = os.path.abspath(os.path.join(output_path, 'SPECULAR/'))

        # Iterate through cameras and render from each one
        for obj in cameras:

            # Specify the output file for each render (adjust as needed)
            comb_path_output = os.path.abspath(os.path.join(combined_file_path_output, obj.name + '.png'))
            
            diff_node = node_tree.nodes.get('File Output')
            spec_node = node_tree.nodes.get('File Output.001')
            diff_node.base_path = os.path.abspath(os.path.join(diffused_file_path_output, obj.name)) 
            spec_node.base_path = os.path.abspath(os.path.join(specular_file_path_output, obj.name)) 

            # Set the output path for the render result
            bpy.context.scene.render.filepath = comb_path_output

            # Set the specified camera as the active camera
            bpy.context.scene.camera = obj       

            # Render the scene
            bpy.ops.render.render(write_still=True)

        fix_dir(output_dir)

        # Close Blender
        bpy.ops.wm.quit_blender()

def fix_dir(path = './PDG/blender_data/'):
    models_dir = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for dir in models_dir:
        model_dir = os.path.abspath(os.path.join(path, dir))
        dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and d != 'COMBINED']
        for sub_dir in dirs:
            edit_dir_path = os.path.abspath(os.path.join(model_dir, sub_dir))
            cams_dir_paths = [d for d in os.listdir(edit_dir_path) if os.path.isdir(os.path.join(edit_dir_path, d))]
            for cam in cams_dir_paths:
                cam_path_folder = os.path.abspath(os.path.join(edit_dir_path, cam))
                img_name = os.path.basename(cam_path_folder) + '.png' 
                cam_path = cam_path_folder + "\Image0001.png"               
                new_path = os.path.join(edit_dir_path, img_name)
                os.rename(cam_path, new_path)
                shutil.rmtree(cam_path_folder)

def get_filepaths(path = './PDG/blender_data/'):
    filepaths = []
    models_dir = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for dir in models_dir:
        model_dir = os.path.abspath(os.path.join(path, dir))
        dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and d != 'SPECULAR']
        for sub_dir in dirs:
            cams_dir_paths = os.path.abspath(os.path.join(model_dir, sub_dir))
            cams = [f for f in os.listdir(cams_dir_paths) if os.path.isfile(os.path.join(cams_dir_paths, f))]
            for cam in cams:
                cam_path_folder = os.path.abspath(os.path.join(cams_dir_paths, cam))
                filepaths.append(cam_path_folder)
    return filepaths

def update_exif_metadata(filepaths):
    # --- update EXIF metadata
    f = config['camera_lens']
    w = config['sensor_width']
    # compute 35mm focal length
    # https://en.wikipedia.org/wiki/35_mm_equivalent_focal_length
    # f35 = 36*f/w [mm]
    fl35 = 36.0 * f / w
    for filepath in filepaths:
        try:
            exiftool_cmd_update = "exiftool -all= -tagsfromfile @ -all:all -unsafe -icc_profile -overwrite_original_in_place " + filepath
            run(exiftool_cmd_update, timeout=5, check=False).returncode
        except Exception as e:
            print(e)  

        exiftool_cmd = [
            "exiftool",
            f"-exif:Make={config['camera']}",
            f"-exif:Model={config['model']}",
            f"-exif:FocalLength={f} mm",
            f"-exif:FocalLengthIn35mmFormat={int(fl35)}",
            "-exif:FocalPlaneXResolution={}".format(config['res_x'] / config['sensor_width']),
            "-exif:FocalPlaneYResolution={}".format(config['res_y'] / config['sensor_height']),
            "-exif:FocalPlaneResolutionUnit#=4",
            "-exif:ExifImageWidth={}".format(config['res_x']),
            "-exif:ExifImageHeight={}".format(config['res_y']), 
            "-overwrite_original_in_place",
            filepath
        ]
        try:
            run(exiftool_cmd, timeout=5, check=False).returncode
        except Exception as e:
            print(e)

def add_exif_metadata(path):
    filepaths = get_filepaths(path)
    update_exif_metadata(filepaths)





def get_subdirectories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def get_files(path):
    return [os.path.abspath(os.path.join(path, f)) for f in os.listdir(path)]

def copy_list_of_paths_in_dir(model, list, destination):
    [shutil.copy2(f, os.path.join(destination, model + "_" + os.path.basename(f).split('_')[-1])) for f in list]

def select_random_directories(directories, test_size=90):
    percent_to_select = (1-test_size)*100
    # Calculate the number of directories to select
    num_to_select = int(len(directories) * percent_to_select / 100)
    # Randomly shuffle the list of directories
    random.shuffle(directories)
    # Select the first 'num_to_select' directories
    selected_directories = directories[:num_to_select]
    # The remaining directories are not selected
    not_selected_directories = directories[num_to_select:]

    return selected_directories, not_selected_directories 


# Function to move and rename files
def move_and_rename_files(source_dir, list_dir, dest_dir):
    os.makedirs(dest_dir)
    os.makedirs(dest_dir + "/COMBINED")
    os.makedirs(dest_dir + "/DIFFUSE")
    os.makedirs(dest_dir + "/SPECULAR")

    c_file_count = 0
    d_file_count = 0
    s_file_count = 0

    for dir in list_dir:
        dir_path = os.path.abspath(os.path.join(source_dir, dir))
        c_path = os.path.join(dir_path, "COMBINED")
        d_path = os.path.join(dir_path, "DIFFUSE")
        s_path = os.path.join(dir_path, "SPECULAR")
        c_files = get_files(c_path)
        d_files = get_files(d_path)
        s_files = get_files(s_path)

        for f in c_files:
            dest_filename = f[:f.rfind('_')+1] + f"{c_file_count}.png"
            dest_filename = dest_filename.split('\\')[-1]
            dest_path = os.path.abspath(os.path.join(os.path.join(dest_dir, "COMBINED"), dest_filename))
            shutil.move(f, dest_path)
            c_file_count += 1
        for f in d_files:
            dest_filename = f[:f.rfind('_')+1] + f"{d_file_count}.png"
            dest_filename = dest_filename.split('\\')[-1]
            dest_path = os.path.abspath(os.path.join(os.path.join(dest_dir, "DIFFUSE"), dest_filename))
            shutil.move(f, dest_path)
            d_file_count += 1 
        for f in s_files:
            dest_filename = f[:f.rfind('_')+1] + f"{s_file_count}.png"
            dest_filename = dest_filename.split('\\')[-1]
            dest_path = os.path.abspath(os.path.join(os.path.join(dest_dir, "SPECULAR"), dest_filename))
            shutil.move(f, dest_path)
            s_file_count += 1           

def select_dir_starting_with(source_dir, prefix):
    return [d for d in next(os.walk(source_dir))[1] if d.startswith(prefix)]

def fix_input_directory(path):
    dir_list = get_subdirectories(path)
    unique_list = []
    [unique_list.append(x.split('_')[0]) for x in dir_list if x.split('_')[0] not in unique_list]
    for prefix in unique_list:
        if os.path.exists(path + prefix):
            shutil.rmtree(path + prefix)
        model_list = select_dir_starting_with(path, prefix)
        dest_dir = path + prefix
        move_and_rename_files(path, model_list, dest_dir)
    for dir_name in dir_list:
        shutil.rmtree(path + dir_name)

def is_chunk_totally_black(chunk):
    import cv2 
    # Convert the chunk to grayscale
    gray_chunk = cv2.cvtColor(chunk, cv2.COLOR_BGR2GRAY)
    non_zero = cv2.countNonZero(gray_chunk)

    # Check if all pixel values are zero
    return non_zero < 27500

def preprocess_image(f_path_c, f_path_d, out_dir_c, out_dir_d):
    import cv2 
    chunk_size = 256
    overlap = 110

    # Load the image
    image_c = cv2.imread(f_path_c)
    image_d = cv2.imread(f_path_d)
    # Get image dimensions
    height, width, _ = image_c.shape

    i = 0
    
    # Iterate over the image with overlap
    for y in range(0, height - chunk_size + 1, overlap):
        for x in range(0, width - chunk_size + 1, overlap):
            chunk_c = image_c[y:y+chunk_size, x:x+chunk_size]
            if not is_chunk_totally_black(chunk_c):
                chunk_d = image_d[y:y+chunk_size, x:x+chunk_size]
                path_c = out_dir_c + os.path.splitext(os.path.basename(f_path_c))[0] + f"_chunk_{i}.png"
                path_d = out_dir_d + os.path.splitext(os.path.basename(f_path_d))[0] + f"_chunk_{i}.png"
                cv2.imwrite(path_c, chunk_c)
                cv2.imwrite(path_d, chunk_d)
                i+=1

def generate_chunks(input_directory = './PDG/dataset/pdg_unet/'):
    path = input_directory + "temp/"
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path + "train/")
        os.makedirs(path + "valid/")
        os.makedirs(path + "train/COMBINED/")
        os.makedirs(path + "train/DIFFUSE/")
        os.makedirs(path + "valid/COMBINED/")
        os.makedirs(path + "valid/DIFFUSE/")

        for folder in get_subdirectories(path):
            dir_path = input_directory + folder
            c_path = os.path.join(dir_path, 'COMBINED')
            d_path = os.path.join(dir_path, 'DIFFUSE')
            for filename in os.listdir(c_path):
                c_file_path = os.path.join(c_path, filename)
                d_file_path = os.path.join(d_path, filename)
                output_directory_c = path + folder + "/COMBINED/"
                output_directory_d = path + folder + "/DIFFUSE/"
                preprocess_image(c_file_path, d_file_path, output_directory_c, output_directory_d)  


def preprocess_image_chunks(chunk_size, f_path_c, f_path_d, f_path_s, out_dir_c, out_dir_d, out_dir_s):
    import cv2
    overlap = 400

    # Load the image
    image_c = cv2.imread(f_path_c)
    image_d = cv2.imread(f_path_d)
    image_s = cv2.imread(f_path_s)

    # Get image dimensions
    height, width, _ = image_c.shape

    i = 0
    
    # Iterate over the image with overlap
    for y in range(0, height - chunk_size + 1, overlap):
        for x in range(0, width - chunk_size + 1, overlap):
            chunk_c = image_c[y:y+chunk_size, x:x+chunk_size]
            if not is_chunk_totally_black(chunk_c):
                chunk_d = image_d[y:y+chunk_size, x:x+chunk_size]
                chunk_s = image_s[y:y+chunk_size, x:x+chunk_size]
                path_c = out_dir_c + os.path.splitext(os.path.basename(f_path_c))[0] + f"_chunk_{i}.png"
                path_d = out_dir_d + os.path.splitext(os.path.basename(f_path_d))[0] + f"_chunk_{i}.png"
                path_s = out_dir_s + os.path.splitext(os.path.basename(f_path_s))[0] + f"_chunk_{i}.png"
                cv2.imwrite(path_c, chunk_c)
                cv2.imwrite(path_d, chunk_d)
                cv2.imwrite(path_s, chunk_s)
                i+=1

def generate_dataset(input_directory = './PDG/blender_data/', output_directory = './PDG/dataset/', generate_unet_dataset=True, generate_pix2pix_dataset=True):
    unet_dataset_dir = './PDG/dataset/unet/'
    pix2pix_dataset_dir = './PDG/dataset/pix2pix/'

    train_folders_unet = ['adidas', 'boot1', 'dwarf', 'elephant', 'guardian', 'grinder', 'pomegranate', 'birdvase', 'madonna', 'ridingboot', 'gator', 'gun', 'spiderman', 'owlvase', 'cabbage', 'conga', 'violin']
    valid_folders = ['kvase', 'teapot', 'toydog', 'baby']
    train_folders_p2p = train_folders_unet + valid_folders
    test_folders = ['bfvase', 'brezel', 'helmet1', 'tractor']
    
    train_folders_unet = [d for prefix in train_folders_unet for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d)) and d.startswith(prefix)]
    valid_folders = [d for prefix in valid_folders for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d)) and d.startswith(prefix)]
    train_folders_p2p = [d for prefix in train_folders_p2p for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d)) and d.startswith(prefix)]
    test_folders = [d for prefix in test_folders for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d)) and d.startswith(prefix)]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory) 

        os.makedirs(output_directory + "/train")
        os.makedirs(output_directory + "/valid")
        os.makedirs(output_directory + "/test")
        os.makedirs(output_directory + "/train/COMBINED")
        os.makedirs(output_directory + "/train/DIFFUSE")
        os.makedirs(output_directory + "/train/SPECULAR")
        os.makedirs(output_directory + "/valid/COMBINED")
        os.makedirs(output_directory + "/valid/DIFFUSE")
        os.makedirs(output_directory + "/valid/SPECULAR")
        os.makedirs(output_directory + "/test/COMBINED")
        os.makedirs(output_directory + "/test/DIFFUSE")
        os.makedirs(output_directory + "/test/SPECULAR")

        for folder in [train_folders_unet, valid_folders, test_folders]:
            for model in folder:
                folder_type = "train" if model in train_folders_unet else ("valid" if model in valid_folders else ("test" if model in test_folders else None))
                model_path = os.path.abspath(os.path.join(input_directory, model))
                images_c = get_files(model_path + "/COMBINED")
                images_d = get_files(model_path + "/DIFFUSE")
                images_s = get_files(model_path + "/SPECULAR")
                copy_list_of_paths_in_dir(model, images_c, output_directory + "/" + folder_type + "/COMBINED")
                copy_list_of_paths_in_dir(model, images_d, output_directory + "/" + folder_type + "/DIFFUSE")
                copy_list_of_paths_in_dir(model, images_s, output_directory + "/" + folder_type + "/SPECULAR")

    if(generate_unet_dataset==True):
        chunk_size = 256
        os.makedirs(unet_dataset_dir)
        os.makedirs(unet_dataset_dir + "/train/COMBINED")
        os.makedirs(unet_dataset_dir + "/train/DIFFUSE")
        os.makedirs(unet_dataset_dir + "/train/SPECULAR")
        os.makedirs(unet_dataset_dir + "/valid/COMBINED")
        os.makedirs(unet_dataset_dir + "/valid/DIFFUSE")
        os.makedirs(unet_dataset_dir + "/valid/SPECULAR")
        os.makedirs(unet_dataset_dir + "/test/COMBINED")
        os.makedirs(unet_dataset_dir + "/test/DIFFUSE")
        os.makedirs(unet_dataset_dir + "/test/SPECULAR")

        for folder in get_subdirectories(output_directory):
            dir_path = output_directory + folder
            c_path = os.path.join(dir_path, 'COMBINED')
            d_path = os.path.join(dir_path, 'DIFFUSE')
            s_path = os.path.join(dir_path, 'SPECULAR')
            for filename in os.listdir(c_path):
                c_file_path = os.path.join(c_path, filename)
                d_file_path = os.path.join(d_path, filename)
                s_file_path = os.path.join(s_path, filename)
                output_directory_c = unet_dataset_dir + folder + "/COMBINED/"
                output_directory_d = unet_dataset_dir + folder + "/DIFFUSE/"
                output_directory_s = unet_dataset_dir + folder + "/SPECULAR/"
                preprocess_image_chunks(chunk_size, c_file_path, d_file_path, s_file_path, output_directory_c, output_directory_d, output_directory_s)

    if(generate_pix2pix_dataset==True):
        chunk_size = 1024
        os.makedirs(pix2pix_dataset_dir)
        os.makedirs(pix2pix_dataset_dir + "/train_A")
        os.makedirs(pix2pix_dataset_dir + "/train_B")
        os.makedirs(pix2pix_dataset_dir + "/train_C")
        os.makedirs(pix2pix_dataset_dir + "/test_A")
        os.makedirs(pix2pix_dataset_dir + "/test_B")
        os.makedirs(pix2pix_dataset_dir + "/test_C")

        for folder in get_subdirectories(output_directory):
            dir_path = output_directory + folder
            c_path = os.path.join(dir_path, 'COMBINED')
            d_path = os.path.join(dir_path, 'DIFFUSE')
            s_path = os.path.join(dir_path, 'SPECULAR')
            for filename in os.listdir(c_path):
                model = filename.rsplit('_', 1)[0]
                folder_type = "train" if model in train_folders_p2p else ("test" if model in test_folders else None)
                c_file_path = os.path.join(c_path, filename)
                d_file_path = os.path.join(d_path, filename)
                s_file_path = os.path.join(s_path, filename)
                output_directory_c = pix2pix_dataset_dir + folder_type + "_A/" 
                output_directory_d = pix2pix_dataset_dir + folder_type + "_B/" 
                output_directory_s = pix2pix_dataset_dir + folder_type + "_C/"
                preprocess_image_chunks(chunk_size, c_file_path, d_file_path, s_file_path, output_directory_c, output_directory_d, output_directory_s)

