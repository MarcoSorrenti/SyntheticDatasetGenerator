import bpy
import os
from ..utils.Utils import get_collection, activate_collection
import math
from mathutils import Euler

class Mesh:
    def __init__(self) -> None:
        #get_collection("pdg_collection", "collection")
        #activate_collection("pdg_collection")
        # self.load_mesh()  <---- SOLO QUESTO

        self.load_mesh_test()

    def load_mesh_test(self):
        def get_subdirectories(path):
            return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        
        directory_path = "./PDG/models_data/"
        
        for dir in get_subdirectories(directory_path):
            subdir_path = os.path.join(directory_path, dir)
            blend_file_path = os.path.abspath(os.path.join(subdir_path, dir.lower() + ".blend"))
            try:
                with bpy.data.libraries.load(blend_file_path, link=True) as (data_from, data_to):
                    # You can choose what to link, for example, "objects", "materials", etc.
                    data_to.objects = data_from.objects

                # Append the linked object(s) to the current scene
                for obj in data_to.objects:
                    bpy.context.collection.objects.link(obj)

                    # Update the scene to reflect the changes
                    bpy.context.view_layer.update()
                    #self.generate_new_blend_file(blend_file)
            except Exception as e:
                print(f"Error loading blend file: {e}")

    def clear_scene(self):
        # Clear all objects from the scene
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.select_by_type(type='EMPTY')
        bpy.ops.object.select_by_type(type='LIGHT')
        bpy.ops.object.select_by_type(type='CAMERA')
        bpy.ops.object.delete()

        # Reset the scene
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def save_blend_file(self, file_name):
        # Set the file path where you want to save the .blend file
        directory_path = "./PDG/models/"
        file_name = file_name + '.blend' 
        blend_file_path_output = os.path.abspath(os.path.join(directory_path, file_name))
        # Save the current blend file
        bpy.ops.wm.save_as_mainfile(filepath=blend_file_path_output)
        print(f"File {file_name} salvato!")

    def move_object_to_origin(self, obj):
        minZ = self.get_min_z_vertex(obj)
        # Move mesh to Z=0
        obj.location[2] = -minZ #np.sign(minZ)*minZ
        bpy.context.view_layer.update()

    def get_min_z_vertex(self, obj):
        mw =  obj.matrix_world
        glob_vertex_coordinates = [ mw @ v.co for v in  obj.data.vertices ] # Global coordinates of vertices
        # Find the lowest Z value amongst the object's verts
        return min( [ co.z for co in glob_vertex_coordinates ] ) 

    def get_max_z_vertex(self, obj):
        mw =  obj.matrix_world
        glob_vertex_coordinates = [ mw @ v.co for v in  obj.data.vertices ] # Global coordinates of vertices
        # Find the lowest Z value amongst the object's verts
        return max( [ co.z for co in glob_vertex_coordinates ] ) 

    def scale_objects(self, obj):
        x_obj,y_obj,z_obj = obj.dimensions
        x_ratio = x_obj#/1
        y_ratio = y_obj#/2
        z_ratio = z_obj
        box_dim = .9
        if x_ratio >= box_dim or y_ratio >= box_dim or z_ratio >= box_dim:
            scale_factor = box_dim/max(x_ratio, y_ratio, z_ratio)
            obj.scale = (scale_factor, scale_factor, scale_factor)
            bpy.context.view_layer.update()
    
    def fix_mesh(self, file):
        obj = bpy.context.scene.objects[-1]
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
        bpy.ops.object.location_clear() # Clear location - set location to (0,0,0)
        if obj.rotation_euler != (0,0,0):
            mat = obj.rotation_euler.to_matrix().to_4x4()
            rot_values = (math.radians(obj.rotation_euler[0]), math.radians(obj.rotation_euler[1]), math.radians(obj.rotation_euler[2]))  # convert degrees to radians
            y_rot = Euler(rot_values)  # default order is XYZ
            # use transformation matrix
            y_mat = y_rot.to_matrix().to_4x4()
            mat = y_mat @ mat
            # update object's euler:
            obj.rotation_euler = mat.to_euler()
            #self.move_object_to_origin(obj)
        self.scale_objects(obj)
        self.save_blend_file(file.split('.')[0].lower())

    
    def load_mesh(self):
        # GET ALL BLEND FILE IN THE PATH
        directory_path = "./PDG/models_data/"
        allowed_formats = ('.blend', '.obj', '.stl')
        if os.path.exists(directory_path):
            files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.lower().endswith(allowed_formats)]
            for file in files:
                file_path = os.path.abspath(os.path.join(directory_path, file))
                #if file.lower().endswith('.obj'):
                    #bpy.ops.wm.obj_import(filepath=file_path)
                if file.lower().endswith('.stl'):
                    self.clear_scene()
                    bpy.ops.import_mesh.stl(filepath=file_path)
                    self.fix_mesh(file)

        else:
            print(f"The specified directory '{directory_path}' does not exist.")