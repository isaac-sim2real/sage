import os
import numpy as np
from dct_processor import DCTMotionProcessor
import torch

def upsample_motion_file(input_file, output_file, original_freq=50, target_freq=100):
    """Upsample a single motion file using DCT."""
    # Load the motion data
    data = np.load(input_file)
    
    # Create DCT processor
    dct_processor = DCTMotionProcessor(original_freq=original_freq, target_freq=target_freq)
    
    # Convert data to torch tensors
    motion_data = {
        "real_dof_positions": torch.from_numpy(data["real_dof_positions"]),
        "real_dof_velocities": torch.from_numpy(data["real_dof_velocities"]),
        "real_dof_positions_cmd": torch.from_numpy(data["real_dof_positions_cmd"]),
        "real_dof_torques": torch.from_numpy(data["real_dof_torques"]),
        "joint_sequence": data["joint_sequence"]
    }
    
    # Process the motion data
    processed_data = dct_processor.process_motion_data(motion_data)
    
    # Convert back to numpy arrays
    output_data = {
        "real_dof_positions": processed_data["real_dof_positions"].numpy(),
        "real_dof_velocities": processed_data["real_dof_velocities"].numpy(),
        "real_dof_positions_cmd": processed_data["real_dof_positions_cmd"].numpy(),
        "real_dof_torques": processed_data["real_dof_torques"].numpy(),
        "joint_sequence": processed_data["joint_sequence"]
    }
    
    # Save the upsampled data
    np.savez(output_file, **output_data)
    print("Processed {} -> {}".format(input_file, output_file))

def process_directory(input_dir, output_dir, original_freq=50, target_freq=100):
    """Process all .npz files in the input directory and save to output directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each .npz file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.npz'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            upsample_motion_file(input_path, output_path, original_freq, target_freq)

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "motions/motion_perjoint_all/edited_27dof"
    output_dir = "motions/motion_perjoint_all/edited_27dof_upsampled"
    
    # Process all files
    process_directory(input_dir, output_dir)