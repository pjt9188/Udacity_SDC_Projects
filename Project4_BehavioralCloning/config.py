"""
Configurations
"""

NVIDIA_H, NVIDIA_W = 66, 200

CONFIG = {
    
    'input_height'              : 160,
    'input_width'               : 320,
    'input_channels'            : 3,
    'delta_correction'          : 0.08,
    'batch_size'                : 1285,

    'augmentation_steer_sigma'  : 0.2,
    'augmentation_value_min'    : 0.2,
    'augmentation_value_max'    : 1.5
}