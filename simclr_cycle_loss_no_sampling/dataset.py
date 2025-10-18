import numpy as np
import torch
from augment_functions import (augment_sample, augment_sample_random_mask,
                               random_mask, resize_encoder)
from torch.utils.data import Dataset


class WeatherBenchDataset(Dataset):
    """
    PyTorch Dataset for WeatherBench data with temporal augmentation and masking.
    """

    def __init__(
        self,
        data,
        max_delta_t=5,
        decay=0.1,
        mask_prob_low=0.5,
        mask_prob_high=0.9,
    ):
        """
        Args:
            data (np.ndarray): Input data array.
            max_delta_t (int): Maximum time delta for augmentation.
            decay (float): Decay rate for delta weighting.
            window (int): Window size for hard negative sampling.
            gap (int): Gap for soft negative sampling.
            mask_prob_low (float): Lower bound for masking probability.
            mask_prob_high (float): Upper bound for masking probability.
        """
        self.data = data
        self.max_delta_t = max_delta_t
        self.mask_prob_low = mask_prob_low
        self.mask_prob_high = mask_prob_high

        self.delta_ts = np.arange(1, max_delta_t + 1)
        self.delta_weights = np.exp(-decay * self.delta_ts)
        self.delta_weights /= self.delta_weights.sum()

    def __len__(self):
        return self.data.shape[0] - (2 * self.max_delta_t)


    def _create_sample(self, idx):
        """
        Create augmented and masked samples for a given index.
        """
        idx = idx + self.max_delta_t
        X = self.data[idx]

        augment_idx = np.random.choice(self.delta_ts, p=self.delta_weights)
        X_prime = self.data[idx + augment_idx]
        X_prime_2 = self.data[idx - augment_idx]

        X_delta = resize_encoder(self.data[idx + augment_idx])
        X_minus_delta = resize_encoder(self.data[idx - augment_idx])

        X_enc = resize_encoder(X)

        X_masked = random_mask(
            X_enc,
            mask_prob_low=self.mask_prob_low,
            mask_prob_high=self.mask_prob_high,
        )

        x = augment_sample(X)
        x_prime = augment_sample_random_mask(
            X_prime,
            mask_prob_low=self.mask_prob_low,
            mask_prob_high=self.mask_prob_high,
        )
        x_prime_2 = augment_sample_random_mask(
            X_prime_2,
            mask_prob_low=self.mask_prob_low,
            mask_prob_high=self.mask_prob_high,
        )

        return (
            x,
            x_prime,
            x_prime_2,
            X,
            X_masked,
            X_prime,
            X_prime_2,
            X_enc,
            X_delta,
            X_minus_delta,
        )

    def __getitem__(self, idx):
        """
        Returns:
            dict: Dictionary of torch.Tensors for augmented and masked samples,
                  with keys labeling anchor, soft negative, and hard negative.
        """
        # Anchor sample
        (
            x,
            x_prime,
            x_prime_2,
            X,
            X_masked,
            X_prime,
            X_prime_2,
            X_enc,
            X_delta,
            X_minus_delta,
        ) = self._create_sample(idx)

        return {
            "x_pos_1": x,
            "x_pos_2": x_prime,
            "x_pos_3": x_prime_2,
            "X_orig": X,
            "X_masked": X_masked,
            "X_masked_delta": X_prime,
            "X_masked_delta_2": X_prime_2,
            "X_enc": X_enc,
            "X_delta": X_delta,
            "X_minus_delta": X_minus_delta,
        }
