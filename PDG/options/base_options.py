import argparse
import sys

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='PDG', help='name of the experiment')        
        self.parser.add_argument('--light_setup', type=str, default='2', help='lights setup: e.g. 1 - use 1,2,3 for multiple configuration')
        self.parser.add_argument('--camera_setup', type=str, default='1', help='cameras setup: e.g. 1 - use 1,2,3 for multiple configuration')
        self.parser.add_argument('--input_blender_directory_path', type=str, default='./PDG/models_data/', help='where the blend object models are located in')
        self.parser.add_argument('--blender_output_dir', type=str, default='./PDG/blender_model_output/', help='output dir for blender object models')
        self.parser.add_argument('--blender_dataset_dir', type=str, default='./PDG/blender_data/', help='directory where the generated images has to be placed')
        
        self.parser.add_argument('--dataset_dir', type=str, default='./PDG/dataset/pdg/', help='directory of the dataset')
        self.parser.add_argument('--unet_dataset_dir', type=str, default='./PDG/dataset/unet/', help='directory of the dataset for the unet')
        self.parser.add_argument('--unet_output_dir', type=str, default='./PDG/results/unet/', help='directory of the predicted images (unet)')
        self.parser.add_argument('--unet_chunk_dataset_dir', type=str, default='./PDG/dataset/pdg_unet_chunk/', help='dataset with chunks')

        # program executions
        self.parser.add_argument('--create_blender_file', type=str, default='True', help='default True. Set to False if you don\'t want to execute the blender file creation process')
        self.parser.add_argument('--render', type=str, default='True', help='default True. Set to False if you don\'t want to execute the rendering process')
        self.parser.add_argument('--add_exif_metadata', type=str, default='False', help='default False. Set to True if you want to add exif metadata')
        self.parser.add_argument('--create_dataset', type=str, default='True', help='default True. Set to False if you don\'t want to execute the dataset creation process')
        self.parser.add_argument('--create_unet_dataset', type=str, default='True', help='default True. Set to False if you don\'t want to execute the dataset creation process for the unet')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args(sys.argv[sys.argv.index("--")+1:])

        args = vars(self.opt)

        print()
        print('------------ Options -------------')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        print()
        return self.opt