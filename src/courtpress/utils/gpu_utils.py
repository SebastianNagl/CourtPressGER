import warnings
from typing import Dict, Any, Tuple


def check_gpu_availability() -> Tuple[bool, Dict[str, Any]]:
    """
    Check if GPU acceleration is available for data processing.

    Returns:
        Tuple containing:
        - Boolean indicating if GPU is available
        - Dictionary with details about the GPU environment
    """
    gpu_info = {
        'libraries': {},
        'memory_pool': False,
        'devices': []
    }

    try:
        # Try to import CUDA libraries
        import cudf
        import cuml
        import cupy as cp

        gpu_info['libraries']['cudf'] = True
        gpu_info['libraries']['cuml'] = True
        gpu_info['libraries']['cupy'] = True

        # Check CUDA version
        gpu_info['cuda_version'] = cp.cuda.runtime.runtimeGetVersion()

        # Check number of GPUs
        num_gpus = cp.cuda.runtime.getDeviceCount()
        gpu_info['num_gpus'] = num_gpus

        # Get device information for each GPU
        for i in range(num_gpus):
            cp.cuda.runtime.setDevice(i)
            device_props = cp.cuda.runtime.getDeviceProperties(i)

            device_info = {
                'id': i,
                'name': device_props['name'].decode(),
                'total_memory': device_props['totalGlobalMem'],
                'compute_capability': f"{device_props['major']}.{device_props['minor']}"
            }

            gpu_info['devices'].append(device_info)

        # Check for RAPIDS memory pool
        try:
            import rmm
            gpu_info['memory_pool'] = True
        except ImportError:
            pass

        # Check NLP-specific GPU support
        try:
            import spacy_cuda
            gpu_info['libraries']['spacy_cuda'] = True
        except ImportError:
            gpu_info['libraries']['spacy_cuda'] = False

        return True, gpu_info

    except ImportError as e:
        # One or more GPU libraries not available
        return False, {'error': str(e)}
    except Exception as e:
        # Other error occurred
        return False, {'error': str(e)}


def setup_gpu_environment() -> Dict[str, Any]:
    """
    Set up the GPU environment with optimized settings if available.

    Returns:
        Dictionary with setup results
    """
    result = {'success': False, 'actions': []}

    # Check if GPU is available
    gpu_available, gpu_info = check_gpu_availability()
    result['gpu_available'] = gpu_available

    if not gpu_available:
        result['message'] = "GPU acceleration not available, using CPU mode"
        warnings.warn(
            "GPU acceleration libraries not found, falling back to CPU mode")
        return result

    # Set up RMM memory pool if available
    if gpu_info.get('memory_pool', False):
        try:
            # Import inside try block to avoid linter errors
            try:
                import rmm
                rmm.reinitialize(managed_memory=True)
                pool = rmm.mr.PoolMemoryResource(
                    rmm.mr.get_current_device_resource())
                rmm.mr.set_current_device_resource(pool)
                result['actions'].append("Initialized RMM memory pool")
            except ImportError:
                result['actions'].append("RMM library not available")
        except Exception as e:
            result['actions'].append(
                f"Failed to initialize RMM memory pool: {e}")

    # Set up cuPy memory pool
    try:
        # Import inside try block to avoid linter errors
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(fraction=0.8)  # Use up to 80% of GPU memory
            result['actions'].append("Configured cuPy memory pool")
        except ImportError:
            result['actions'].append("cuPy library not available")
    except Exception as e:
        result['actions'].append(f"Failed to configure cuPy memory pool: {e}")

    result['success'] = True
    result['message'] = "GPU environment set up successfully"
    return result


def cleanup_gpu_memory() -> None:
    """
    Free unused GPU memory to prevent memory fragmentation and leaks.
    """
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        # Free all unused memory blocks
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        # Print memory usage info
        print(f"GPU memory pool: {mempool.used_bytes()} bytes used")
    except ImportError:
        # cuPy not available, nothing to do
        pass
    except Exception as e:
        warnings.warn(f"Error cleaning up GPU memory: {e}")
