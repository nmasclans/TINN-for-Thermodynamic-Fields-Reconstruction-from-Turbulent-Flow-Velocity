import tensorflow as tf
import numpy as np

from datetime import datetime

def tf_print_time():
    now = datetime.now().strftime("%H:%M:%S")
    tf.print("Current Time:",now)

def transform_targets_to_original_scaling(args, *multiple_targets):
    """
    Converts input tensors from scaled range to original range
    Args:
        args: my_parser arguments
        multiple_targets (iteable of tf.tensor): tf tensor, of shape = (batch_size,num_targets)
    Returns:
        tuple of np arrays of targets rescaled to the original scaling
    """
    def _targets_to_original_scaling(args, targets):
        targets_transformed = np.zeros_like(targets)
        for targ_idx in range(args.num_targets):
            targ_name = args.targets_name[targ_idx]
            targ_min  = args.targets_limits[targ_name][0]
            targ_max  = args.targets_limits[targ_name][1]
            assert (targ_max-targ_min) > 0
            targets_transformed[:,targ_idx] = (targets[:,targ_idx] - args.min_value) * (targ_max-targ_min) / (args.max_value-args.min_value) + targ_min 
        return targets_transformed

    return (_targets_to_original_scaling(args, t) for t in multiple_targets)

def classify_state_from_temperature(args, *multiple_T_tensors):
    """
    Transforms temperature data (continuous data) to fluid state (discrete, 3 classes) 
    Args:
        args: my_parser arguments
        multiple_T_tensors (iteable of tf.tensor): tf.tensor / np.array of temperature values, shape = (batch_size,)
    return:
        tuple of np arrays of fluid states information, shape = (batch_size,), where:
            0 : liquid-like       (T < args.T_minus)
            1 : two-phases-like   (args.T_minus < T < args.T_plus)
            2 : gas-like          (args.T_plus < T)
    """
    def _state_from_temperature(args, T):
        T_minus = args.T_minus
        T_plus  = args.T_plus
        state = np.ones(len(T),dtype=int)
        state[T < T_minus] = 0
        state[T > T_plus]  = 2
        return state

    return (_state_from_temperature(args, T_tensor) for T_tensor in multiple_T_tensors)