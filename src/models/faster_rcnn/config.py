"""Faster R-CNN default configuration."""

from typing import Dict, Tuple


def get_default_config() -> Dict:
    """
    Get default configuration for Faster R-CNN.

    Returns:
        Configuration dictionary
    """
    return {
        # Image preprocessing
        'min_size': 384,
        'max_size': 384,

        # Anchor generation
        'anchor_sizes': ((32, 64, 128, 256, 512),),
        'anchor_aspect_ratios': ((0.5, 1.0, 2.0),),

        # ROI pooling
        'roi_output_size': 7,
        'roi_sampling_ratio': 2,

        # RPN parameters
        'rpn_pre_nms_top_n_train': 1000,
        'rpn_pre_nms_top_n_test': 500,
        'rpn_post_nms_top_n_train': 500,
        'rpn_post_nms_top_n_test': 250,
        'rpn_batch_size_per_image': 32,
        'rpn_fg_iou_thresh': 0.7,
        'rpn_bg_iou_thresh': 0.3,

        # Detection parameters
        'box_detections_per_img': 100,
        'box_score_thresh': 0.01,
        'box_nms_thresh': 0.3,
    }
