# Synthetic Dataset Generation for Photogrammetric 3D Reconstruction

This software facilitates the generation of synthetic datasets to simulate the photogrammetric acquisition process for subsequent 3D reconstruction of objects. By leveraging this tool, users can efficiently create synthetic data simulating the real-world scenarios encountered in photogrammetry, enabling comprehensive testing and validation of algorithms and systems involved in 3D reconstruction processes.

## Features

- **Customizable Parameters**: Users can adjust various parameters such as lighting conditions, camera settings, object properties, and environmental factors to tailor the synthetic dataset according to specific research or testing requirements.
  
- **Realistic Simulation**: The software employs algorithms to simulate the photogrammetric acquisition process realistically, including camera viewpoints and lighting setup ensuring that the generated dataset closely resembles real-world scenarios.
  
- **Scalability**: generate datasets for small-scale objects, the software offers scalability to accommodate diverse applications within the field of photogrammetry and 3D reconstruction.
  
- **Export Formats**: generated datasets can be used in various standard formats compatible with popular photogrammetry and computer vision software, promoting interoperability and seamless integration into existing workflows.

## Usage

- **3D models**: add 3D models into a directory of your choice
- **Setup**: change parameters with respect to your specific task in the file option/base_option.py
- **How to Run**: by using CMD, use the following prompt: "blender -b --python preprocessing.py" that allow to execute blender in background.
