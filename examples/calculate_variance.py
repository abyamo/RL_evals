import numpy as np

def calculate_reward_variance(reward_vector):
    """
    Calculate the variance of a list or numpy array of rewards.
    
    Args:
        reward_vector (list or np.ndarray): List or array of scalar rewards.
    
    Returns:
        float: Variance of the rewards.
    """
    reward_array = np.array(reward_vector)
    variance = np.var(reward_array)
    return variance