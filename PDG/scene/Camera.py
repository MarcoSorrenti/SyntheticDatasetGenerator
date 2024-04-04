import bpy
import math, mathutils
import numpy as np

from ..utils.Utils import activate_collection, _fibonacci_pt, rotate

class Camera:
    def __init__(self, h_dim, object, model = "CanonR6_MARKII") -> None:
        self._h_dim = h_dim
        self._object = object
        self._model_name = model
        self.lens = None
        self.sensor_fit = "HORIZONTAL"
        self.sensor_width = None
        self.sensor_height = None
        self.set_camera_model()

        self.add_camera_setup_1()
        self.add_camera_setup_2()
        #self.add_camera_setup_3()
        #self.add_camera_setup_4()

    def set_up_lens(self, sens_width, sens_lenght, lens):
        self.lens = lens
        self.sensor_width = sens_width
        self.sensor_height = sens_lenght

    def set_camera_model(self):
        if self._model_name == "Canon6D35":
            # https://www.digicamdb.com/specs/canon_eos-6d/
            self._brand_name = "Canon"
            self.set_up_lens(35.8, 23.9, 35)
        elif self._model_name == "Canon6D24":
            self._brand_name = "Canon"
            self.set_up_lens(35.8, 23.9, 24)
        elif self._model_name == "Canon6D14":
            self._brand_name = "Canon"
            self.set_up_lens(35.8, 23.9, 14.46)
        elif self._model_name == "Canon6D_MARKII_35":
            # https://www.digicamdb.com/specs/canon_eos-6d-mark-ii/
            self._brand_name = "Canon"
            self.set_up_lens(35.9, 24, 35)
        elif self._model_name == "CanonR6_MARKII":
            # https://www.digicamdb.com/specs/canon_r6-mark-ii/
            self._brand_name = "Canon"
            self.set_up_lens(35.9, 23.9, 35)
        elif self._model_name == "CanonR7":
            # https://www.digicamdb.com/specs/canon_r7/
            self._brand_name = "Canon"
            self.set_up_lens(22.3, 14.9, 35)

    def set_camera(self, camera_obj):
        camera_obj.lens = self.lens
        camera_obj.sensor_fit = self.sensor_fit
        camera_obj.sensor_width = self.sensor_width
        camera_obj.sensor_height = self.sensor_height


    # SETUP 1: semisfera
    def add_camera_setup_1(self, number_of_cameras = 80):
        activate_collection('camera_setup_1')
        
        focus_point=mathutils.Vector((0.0, 0.0, 0.0))
        cam_locations = []

        for i in range(number_of_cameras):
            fib_loc_i = [_fibonacci_pt(1.2, i, number_of_cameras)]
            if fib_loc_i[0][2] > 0:
                fib_loc_i[0] = (fib_loc_i[0][0], fib_loc_i[0][1], fib_loc_i[0][2])
                cam_locations.append(fib_loc_i)
                
        cam_locations =  [cam_location for cam_location in cam_locations if cam_location[0][2]>0]

        #DINAMIC PARAMS
        scale_factor = self._h_dim * 1.5

        # Add cameras
        for idx, i in enumerate(cam_locations):
            bpy.ops.object.camera_add(location=i[0], rotation=(0, 0, 0))
            camera_obj = bpy.context.scene.objects[-1]
            camera_obj.name = "Cam_s1_"+ str(idx)
            looking_direction = camera_obj.location - focus_point
            rot_quat = looking_direction.to_track_quat('Z', 'Y')
            camera_obj.rotation_euler = rot_quat.to_euler()
            camera_obj.location = rot_quat @ mathutils.Vector((0.0, 0.0, scale_factor))
            #camera_obj.location.z = camera_obj.location.z + self._object.dimensions.z*1.2  #only for the model: GUN
            camera_obj.location.z = camera_obj.location.z + self._object.dimensions.z/2
            # Get the current rotation in Euler angles
            rot_euler = camera_obj.rotation_euler
            rot_euler.x -= math.radians(self._h_dim)
            camera_obj.rotation_euler = rot_euler
            self.set_camera(camera_obj.data)

        #print("Aggiunto setup camera 1: semisfera")
        
    # SETUP 2: cilindro + sfera
    def add_camera_setup_2(self):
        activate_collection('camera_setup_2')
        focus_point=mathutils.Vector((0.0, 0.0, 0.0))

        z_step = (self._h_dim+(.1*self._h_dim))/3.
        # z_position computed using high of the tracked object (3 circles on obj z axis)
        z_positions = [ z_step, 2*z_step, 3*z_step]
        # rotation on diagonal axis    
        cam_locations = []
            
        # Cylinder disposition
        ANGLE_STEP = 20 # 20
        angle_steps = int(360/ANGLE_STEP)
        cam_location = np.array([1,0,z_positions[0]])
        for z_pos in z_positions:
            cam_location = np.array([cam_location[0], cam_location[1], z_pos])
            for idx_axis, axis in enumerate([(0,0,1)]):
                for i in range(1, angle_steps+1):
                    cam_location = rotate(cam_location, ANGLE_STEP, axis=axis)
                    cam_locations.append(np.round(cam_location,2))
    
        # return only positive z camera positions
        cam_locations =  [cam_location for cam_location in cam_locations if cam_location[2]>0]

        #DINAMIC PARAMS
        scale_factor = self._h_dim * 3

        # Add cameras
        for idx, i in enumerate(cam_locations):
            bpy.ops.object.camera_add(location=i, rotation=(0, 0, 0))
            camera_obj = bpy.context.scene.objects[-1]
            camera_obj.name = "Cam_s2_"+ str(idx)
            looking_direction = camera_obj.location - focus_point
            rot_quat = looking_direction.to_track_quat('Z', 'Y')

            camera_obj.rotation_euler = rot_quat.to_euler()
            camera_obj.location = rot_quat @ mathutils.Vector((0.0, 0.0, scale_factor))
            camera_obj.location.z = camera_obj.location.z + self._object.dimensions.z/2
            self.set_camera(camera_obj.data)

        #print("Aggiunto setup camera 2: cilindro + sfera")        

    def add_camera_setup_3(self, number_of_cameras = 100):
        activate_collection('camera_setup_3')
        golden_angle = np.pi * (3 - np.sqrt(5))
        theta = golden_angle * np.arange(number_of_cameras)
        z = np.linspace(1 - 1.0 / number_of_cameras, 1.0 / number_of_cameras - 1, number_of_cameras)
        radius = np.sqrt(1 - z * z)
            
        points = np.zeros((number_of_cameras, 3))
        points[:,0] = radius * np.cos(theta)
        points[:,1] = radius * np.sin(theta)
        points[:,2] = z

        focus_point = mathutils.Vector((0.0, 0.0, 0.0))

        # Add cameras
        for idx, i in enumerate(points):

            bpy.ops.object.camera_add(location=i, rotation=(0, 0, 0))
            camera_obj = bpy.context.scene.objects[-1]
            camera_obj.name = "Cam_s3_"+ str(idx)
            looking_direction = camera_obj.location - focus_point
            rot_quat = looking_direction.to_track_quat('Z', 'Y')

            camera_obj.rotation_euler = rot_quat.to_euler()
            camera_obj.location = rot_quat @ mathutils.Vector((0.0, 0.0, 10.0))
        print("Aggiunto setup camera 3: sfera")

    def add_camera_setup_4(self, number_of_cameras = 10, cylinder_radius = 5.0, cylinder_height = 2.0):
        activate_collection('camera_setup_4')

        focus_point=mathutils.Vector((0.0, 0.0, 0.0))
        # Add cameras
        for i in range(number_of_cameras):
            angle = (2 * math.pi * i) / number_of_cameras
            x = cylinder_radius * math.cos(angle)
            y = cylinder_radius * math.sin(angle)

            bpy.ops.object.camera_add(location=(x, y, cylinder_height), rotation=(0, 0, angle))
            camera_obj = bpy.context.scene.objects[-1]
            camera_obj.name = "Cam_s4_"+ str(i)
            looking_direction = camera_obj.location - focus_point
            rot_quat = looking_direction.to_track_quat('Z', 'Y')

            camera_obj.rotation_euler = rot_quat.to_euler()
            # Use * instead of @ for Blender <2.8
            camera_obj.location = rot_quat @ mathutils.Vector((0.0, 0.0, 10.0))

        print("Aggiunto setup camera 4: circolare lungo un piano")