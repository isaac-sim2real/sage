import numpy as np
import torch
from scipy.fft import dct, idct

class DCTMotionProcessor:
    """Process motion data using Discrete Cosine Transform (DCT) for upsampling."""
    
    def __init__(self, original_freq=50, target_freq=100):
        """Initialize the DCT processor.
        
        Args:
            original_freq (int): Original motion frequency (Hz)
            target_freq (int): Target motion frequency (Hz)
        """
        self.original_freq = original_freq
        self.target_freq = target_freq
        self.upsample_ratio = target_freq // original_freq
        
    def upsample_motion(self, motion_data: torch.Tensor) -> torch.Tensor:
        """Upsample motion data using DCT.
        
        Args:
            motion_data (torch.Tensor): Input motion data of shape (num_frames, num_dofs)
            
        Returns:
            torch.Tensor: Upsampled motion data of shape (num_frames * upsample_ratio, num_dofs)
        """
            
        # Convert to numpy for DCT processing
        motion_np = motion_data.cpu().numpy()
        num_frames, num_features, num_dofs = motion_np.shape
        
        # Initialize output array
        upsampled_motion = np.zeros((num_frames * self.upsample_ratio, num_features, num_dofs))
                
        # Process each feature and DOF separately
        for feature in range(num_features):
            for dof in range(num_dofs):
                # Get the motion sequence for this feature and DOF
                motion_seq = motion_np[:, feature, dof]
                
                # Apply DCT
                dct_coeffs = dct(motion_seq, type=2, norm='ortho')
                
                # Zero-pad the DCT coefficients
                padded_coeffs = np.zeros(len(dct_coeffs) * self.upsample_ratio)
                padded_coeffs[:len(dct_coeffs)] = dct_coeffs
                
                # Apply inverse DCT
                upsampled_seq = idct(padded_coeffs, type=2, norm='ortho')
                
                # Store the upsampled sequence
                upsampled_motion[:, feature, dof] = upsampled_seq
            
        # Convert back to torch tensor
        return torch.from_numpy(upsampled_motion).to(motion_data.device)
    
    def process_motion_data(self, motion_data: dict) -> dict:
        """Process all motion data arrays in the dictionary.
        
        Args:
            motion_data (dict): Dictionary containing motion data arrays
            
        Returns:
            dict: Dictionary containing upsampled motion data arrays
        """
        processed_data = {}
        
        # Process each array except joint_sequence
        for key, value in motion_data.items():
            if key != "joint_sequence":
                if isinstance(value, torch.Tensor):
                    processed_data[key] = self.upsample_motion(value)
                else:
                    processed_data[key] = self.upsample_motion(torch.from_numpy(value))
            else:
                processed_data[key] = value
                
        return processed_data