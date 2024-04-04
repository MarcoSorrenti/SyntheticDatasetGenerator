import bpy
import os
from .Light import Light
from .Camera import Camera

from ..utils.Utils import get_collection, activate_collection, l_c_setup

class Scene:
    def __init__(self, object, l_setup, c_setup) -> None:
        self._context = bpy.context
        self._scene = self._context.scene       
        self._scene.render.image_settings.file_format='PNG'
        self._scene.render.resolution_x = 2000
        self._scene.render.resolution_y = 2000
        self._object = object
        self._h_dim = max(self._object.dimensions)
        self._light_setup = l_setup
        self._camera_setup = c_setup

    def create_collections(self):
        self.environment_collection = get_collection("pdg_collection", "collection")
        
        self.light_collection = get_collection("light_collection")
        self.camera_collection = get_collection("camera_collection")
        self.environment_collection.children.link(self.light_collection)
        self.environment_collection.children.link(self.camera_collection)

        self.light_collection.children.link(get_collection("light_ground"))
        self.light_collection.children.link(get_collection("light_setup_1"))
        self.light_collection.children.link(get_collection("light_setup_2"))
        self.light_collection.children.link(get_collection("light_setup_3"))
        self.camera_collection.children.link(get_collection("camera_setup_1"))
        self.camera_collection.children.link(get_collection("camera_setup_2"))

    def add_floor(self):
        activate_collection('pdg_collection')

        floor_size = self._h_dim * 2.5
        # Create a cube
        bpy.ops.mesh.primitive_cube_add(size=floor_size, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        floor = bpy.context.scene.objects[-1]

        new_height = floor_size * 0.8 / 10

        # Move the cube below the origin
        floor.dimensions.z = new_height
        # Move the cube below the origin
        floor.location.z = -new_height / 2
        #Rename it
        floor.name = "Floor"

        #ADD MATERIAL (OPTIONAL)
        # Create a new material
        material = bpy.data.materials.new(name="BlackMaterial")
        # Make the material completely black
        material.use_nodes = False  # Disable nodes for simplicity
        material.diffuse_color = (0, 0, 0, 1)  # Set the diffuse color to black
        floor.data.materials.append(material)

    def generate_new_blend_file(self, file_name):
        # Set the file path where you want to save the .blend file
        directory_path = "./PDG/output/"
        blend_file_path_output = os.path.abspath(os.path.join(directory_path, file_name))
  
        # Save the current blend file
        bpy.ops.wm.save_as_mainfile(filepath=blend_file_path_output)
        print(f"File {file_name} salvato!")
  
    def add_background(self):
        world = bpy.data.worlds.new(name="World")

        if not world.use_nodes:
            world.use_nodes = True

        nodes = world.node_tree.nodes

        # Clear Existing nodes
        for current_node in nodes:
            world.node_tree.nodes.remove(current_node)

        world_output = nodes.new('ShaderNodeTexEnvironment')
    
    def prepare_scene(self):

        def get_collections_recursive(layer_collection):
            result = [layer_collection.name]
            
            for child_collection in layer_collection.children:
                result.extend(get_collections_recursive(child_collection))

            return result
        
        # Get the top-level layer collection
        top_layer_collection = bpy.context.view_layer.layer_collection

        # Get collections recursively
        all_collections = get_collections_recursive(top_layer_collection)

        # Add the names of layer collections you want to exclude
        include_layer_names = ["pdg_collection", "light_setup_2", "camera_setup_2"]

        excluded_layer_names = list(filter(lambda x: x not in include_layer_names, all_collections))

        def exclude_layer_collection_recursive(layer_collection):
            if layer_collection.name in excluded_layer_names:
                layer_collection.exclude = True

            for child_collection in layer_collection.children:
                exclude_layer_collection_recursive(child_collection)

        # Exclude layer collections recursively
        exclude_layer_collection_recursive(top_layer_collection)

        def include_view_layer(name: str = "pdg_collection"):

            # Get the top-level layer collection
            top_layer_collection = bpy.context.view_layer.layer_collection

            def find_layer_collection_recursive(layer_collection):
                if layer_collection.name == name:
                    layer_collection.exclude = False

                for child_collection in layer_collection.children:
                    find_layer_collection_recursive(child_collection)

            # Exclude layer collections recursively
            find_layer_collection_recursive(top_layer_collection)

        #for layer in include_layer_names:
        for key, value in l_c_setup['light'].items():
            if value == self._light_setup:
                include_view_layer(key)

        for key, value in l_c_setup['camera'].items():
            if value == self._camera_setup:
                include_view_layer(key)

        include_view_layer("light_ground")

    def render_scene(self):
        self.create_collections()
        #self.add_floor()
        #self.add_background()
        self._light = Light(self._h_dim)  
        self._camera = Camera(self._h_dim, self._object)

        self.prepare_scene()