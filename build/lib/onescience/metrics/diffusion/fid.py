import torch

from onescience.metrics.general.wasserstein import wasserstein_from_normal


def calculate_fid_from_inception_stats(
    mu: torch.Tensor, sigma: torch.Tensor, mu_ref: torch.Tensor, sigma_ref: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the Fréchet Inception Distance (FID) between two sets
    of Inception statistics.

    The Fréchet Inception Distance is a measure of the similarity between two datasets
    based on their Inception features (mu and sigma). It is commonly used to evaluate
    the quality of generated images in generative models.

    Parameters
    ----------
    mu:  torch.Tensor:
        Mean of Inception statistics for the generated dataset.
    sigma: torch.Tensor:
        Covariance matrix of Inception statistics for the generated dataset.
    mu_ref: torch.Tensor
        Mean of Inception statistics for the reference dataset.
    sigma_ref: torch.Tensor
        Covariance matrix of Inception statistics for the reference dataset.

    Returns
    -------
    float
        The Fréchet Inception Distance (FID) between the two datasets.
    """
    return wasserstein_from_normal(mu, sigma, mu_ref, sigma_ref)
