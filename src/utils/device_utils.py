#!/usr/bin/env python3
"""
Device utilities for handling GPU/CPU device selection.

This module provides utilities to parse and validate device arguments,
supporting both traditional options and specific GPU ID selection.
"""

import torch
import logging
import re
from typing import Union

logger = logging.getLogger(__name__)


def parse_device(device_arg: str) -> torch.device:
    """
    Parse device argument and return appropriate torch.device.

    Supported formats:
    - 'cpu': Use CPU
    - 'cuda': Use default CUDA device
    - 'auto': Auto-select best available device
    - '0', '1', '2', '3': Use specific GPU ID (converted to cuda:0, cuda:1, etc.)
    - 'cuda:0', 'cuda:1', etc.: Use specific GPU ID

    Args:
        device_arg: Device specification string

    Returns:
        torch.device: Configured device object

    Raises:
        ValueError: If device format is invalid or unavailable
    """
    device_arg = device_arg.lower().strip()

    # Handle basic cases
    if device_arg == 'cpu':
        logger.info("Using CPU device")
        return torch.device('cpu')

    elif device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Auto-selected CUDA device: {device}")
        else:
            device = torch.device('cpu')
            logger.info("CUDA not available, using CPU")
        return device

    elif device_arg == 'cuda':
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available, falling back to CPU")
            return torch.device('cpu')
        device = torch.device('cuda')
        logger.info(f"Using default CUDA device: {device}")
        return device

    # Handle numeric GPU IDs (e.g., '0', '1', '2')
    elif device_arg.isdigit():
        gpu_id = int(device_arg)
        return _validate_and_create_cuda_device(gpu_id)

    # Handle cuda:X format
    elif device_arg.startswith('cuda:'):
        match = re.match(r'cuda:(\d+)', device_arg)
        if match:
            gpu_id = int(match.group(1))
            return _validate_and_create_cuda_device(gpu_id)
        else:
            raise ValueError(f"Invalid CUDA device format: {device_arg}")

    else:
        raise ValueError(f"Unsupported device format: {device_arg}. "
                         f"Supported formats: 'cpu', 'cuda', 'auto', '0', '1', '2', 'cuda:0', 'cuda:1', etc.")


def _validate_and_create_cuda_device(gpu_id: int) -> torch.device:
    """
    Validate GPU ID and create CUDA device.

    Args:
        gpu_id: GPU device ID

    Returns:
        torch.device: Configured CUDA device

    Raises:
        ValueError: If GPU ID is invalid or unavailable
    """
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this system")

    device_count = torch.cuda.device_count()
    if gpu_id >= device_count:
        raise ValueError(
            f"GPU {gpu_id} not available. Available GPUs: 0-{device_count-1}")

    device = torch.device(f'cuda:{gpu_id}')

    # Test if device is actually accessible
    try:
        torch.cuda.set_device(gpu_id)
        # Simple memory check
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_properties(gpu_id).name
        gpu_memory = torch.cuda.get_device_properties(
            gpu_id).total_memory / (1024**3)
        logger.info(f"Using GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f}GB)")
    except Exception as e:
        raise ValueError(f"Cannot access GPU {gpu_id}: {e}")

    return device


def get_device_info(device: torch.device) -> dict:
    """
    Get detailed information about the specified device.

    Args:
        device: torch.device object

    Returns:
        dict: Device information including name, memory, etc.
    """
    info = {
        'type': device.type,
        'index': device.index,
        'name': str(device)
    }

    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(device)
            info.update({
                'gpu_name': props.name,
                'total_memory_gb': props.total_memory / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count
            })

            # Current memory usage
            if torch.cuda.current_device() == device.index:
                info.update({
                    'allocated_memory_gb': torch.cuda.memory_allocated(device) / (1024**3),
                    'reserved_memory_gb': torch.cuda.memory_reserved(device) / (1024**3)
                })
        except Exception as e:
            logger.warning(f"Could not get detailed GPU info: {e}")

    return info


def list_available_devices() -> list:
    """
    List all available devices on the system.

    Returns:
        list: List of available device strings
    """
    devices = ['cpu', 'auto']

    if torch.cuda.is_available():
        devices.append('cuda')
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            devices.extend([str(i), f'cuda:{i}'])

    return devices


def get_optimal_device() -> torch.device:
    """
    Get the optimal device based on system capabilities.

    Returns:
        torch.device: Best available device
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')

    # Find GPU with most available memory
    device_count = torch.cuda.device_count()
    if device_count == 1:
        return torch.device('cuda:0')

    best_gpu = 0
    max_memory = 0

    for i in range(device_count):
        try:
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory

            # Try to get current memory usage
            current_device = torch.cuda.current_device()
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i)
            available_memory = total_memory - allocated
            torch.cuda.set_device(current_device)  # Reset

            if available_memory > max_memory:
                max_memory = available_memory
                best_gpu = i

        except Exception:
            continue

    return torch.device(f'cuda:{best_gpu}')


# For backward compatibility
def validate_device_argument(device_str: str) -> str:
    """
    Validate device argument string format.

    Args:
        device_str: Device string to validate

    Returns:
        str: Validated device string

    Raises:
        ValueError: If device format is invalid
    """
    try:
        parse_device(device_str)
        return device_str
    except ValueError as e:
        raise ValueError(str(e))
