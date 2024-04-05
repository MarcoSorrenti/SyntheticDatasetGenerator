import bpy
import math

from ..utils.Utils import activate_collection


class Light:
    def __init__(self, dim=1) -> None:
        self._dim = dim
        self.add_ground_light()
        self.add_light_setup_1()
        self.add_light_setup_2()
        self.add_light_setup_3()

    def add_ground_light(self):
        activate_collection('light_ground')
        loc = (0,0,-1.5)
        bpy.ops.object.light_add(type='AREA', radius=1, align='WORLD', location=loc, rotation=(0.0, math.radians(180), 0.0))
        ground_light = bpy.context.scene.objects[-1]
        ground_light.data.energy = 7
        ground_light.data.use_shadow = False
        ground_light.data.specular_factor = 0
        ground_light.data.volume_factor = 0
        ground_light.data.shape = "SQUARE"
        ground_light.data.size = 6
        ground_light.name = "GroundLight"
    
    # SETUP 1: 1 spot light 
    def add_light_setup_1(self):
        activate_collection('light_setup_1')

        #DINAMIC PARAMS
        new_high = self._dim * 2.5
        new_loc = new_high / 2
        loc = (new_loc, -new_loc, new_high)
        
        # Add a spot light
        bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=loc)
        spot_light = bpy.context.scene.objects[-1]

        # Set spot light properties
        spot_light.data.energy = 5  # 10 is the max value for energy
        spot_light.data.angle  = 0.1
        spot_light.rotation_euler = (math.radians(25), math.radians(30), 0)  # Adjust as needed

        # Rename the spot light
        spot_light.name = "SpotLight"
        #print("Aggiunto setup luci 1: One Single SpotLight")

    # SETUP 2: 4 Area Lights square
    def add_light_setup_2(self):
        activate_collection('light_setup_2')

        #DINAMIC PARAMS
        new_high = self._dim * 2.5
        new_loc = new_high / 2
        locations = [(new_loc, new_loc, new_high), (new_loc, -new_loc, new_high), (-new_loc, -new_loc, new_high), (-new_loc, new_loc, new_high)] 
        
        for i, loc in enumerate(locations):
            # Add a spot light
            bpy.ops.object.light_add(type='AREA', radius=1, align='WORLD', location=loc)
            area_light = bpy.context.scene.objects[-1]
            area_light.color = (1,1,1,1)
            area_light.data.energy = 25

            #DINAMIC PARAMS
            area_light.scale.x = self._dim * 2.5 / 10 # Adjust the scale factor as needed

            # Rename the spot light
            area_light.name = "AreaLamp_"+ str(i)
        #print("Aggiunto setup luci 2: 4 Area Lights square!")


    # SETUP 3: 3 Area Lights triangle
    def add_light_setup_3(self):
        activate_collection('light_setup_3')

        #DINAMIC PARAMS
        R = self._dim * 2.5 / 2
        H = self._dim * 2.5
        locations = [(0, R, H), (-R*math.sqrt(3)/2, -R/2, H), (R*math.sqrt(3)/2, -R/2, H)] 
        for i, loc in enumerate(locations):
            bpy.ops.object.light_add(type='AREA', radius=1, align='WORLD', location=loc)
            area_light = bpy.context.scene.objects[-1]
            area_light.location = loc
            area_light.color = (1,1,1,1)
            area_light.data.energy = 50 

            # Rename the spot light
            area_light.name = "AreaLight_"+ str(i)
        #print("Aggiunto setup luci 3: 3 Area Lights triangle")



