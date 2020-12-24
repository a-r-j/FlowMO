import argparse
from typing import Union, Optional, Callable, Tuple, Any, Dict
import gpflow
from gpflow.mean_functions import Constant
import numpy as np
from sklearn.model_selection import train_test_split
import logging as log

# from data_utils import transform_data
from data_utils import transform_data, TaskDataLoader, featurise_mols


class ActiveLearner:
    def __init__(
        self,
        kernel_params: Dict[str, Any],
        model_params: Dict[str, Any],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        acquisition_function: str,
        scaler: str,
        seed: int,
        init_size: float,
        acquisition_size: float,
    ):
        self.kernel_params = kernel_params
        self.model_params = model_params
        self.seed = seed

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.init_size = init_size
        self.acquisition_size = acquisition_size
        self.n_feats = self.x_train.shape[1]
        self.n_samp = self._get_n_samp()

        self.acquisition_function = self._select_acquisition_function(
            acquisition_function
        )
        # self.scaler = self._get_data_scaler(scaler)

        self.optimizer = gpflow.optimizers.Scipy()

    def train(self, n_iter: int):

        X_holdout, X_init, y_holdout, y_init = self.create_initial_sample(
            self.x_train, self.y_train
        )

        for i in range(n_iter + 1):
            log.info("Performing iteration {i}.")
            # Fit Scaler to data
            (
                X_init_scaled,
                y_init_scaled,
                X_test_scaled,
                y_test_scaled,
                y_scaler,
            ) = transform_data(X_init, y_init, self.x_test, self.y_test)
            # todo BUILD MODEL m = self.assemble_model()
            k = gpflow.kernels.Matern32(lengthscales=np.ones(self.n_feats), variance=1)

            m = gpflow.models.GPR(
                data=(X_init_scaled, y_init_scaled),
                kernel=k,
                noise_variance=0.01,
                mean_function=Constant(np.mean(y_init_scaled)),
            )

            # Fit model
            opt_logs = self.optimizer.minimize(
                m.training_loss, m.trainable_variables, options=dict(maxiter=100)
            )

            # Todo Performance logging

            # Update datasets
            (
                X_init_scaled,
                y_init_scaled,
                X_holdout_scaled,
                y_holdout_scaled,
                y_scaler,
            ) = transform_data(X_init, y_init, X_holdout, y_holdout)

            sample_indices = self.suggest_sample(X_holdout_scaled, m, self.n_samp)
            X_init, X_holdout, y_init, y_holdout = self.update_training_data(
                sample_indices, X_init, X_holdout, y_init, y_holdout
            )

    def create_initial_sample(
        self, x_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates initialisation and hold out sets
        """
        X_holdout, X_init, y_holdout, y_init = train_test_split(
            x_train, y_train, test_size=self.init_size, random_state=self.seed
        )
        return X_holdout, X_init, y_holdout, y_init

    def build_model(self):
        raise NotImplementedError

    def suggest_sample(self, A: np.ndarray, model, n_samp: int):
        scores = self.acquisition_function(A, model)
        indices = np.argpartition(scores, -n_samp)[-n_samp:]  # Todo random
        return indices

    def objective_closure(self):
        return -self.model.log_marginal_likelihood()

    @staticmethod
    def update_training_data(
        sample_indices: np.ndarray,
        X_init: np.ndarray,
        X_holdout: np.ndarray,
        y_init: np.ndarray,
        y_holdout: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        X_init = np.concatenate((X_init, X_holdout[sample_indices]))
        y_init = np.concatenate((y_init, y_holdout[sample_indices]))

        X_holdout = np.delete(X_holdout, sample_indices, axis=0)
        y_holdout = np.delete(y_holdout, sample_indices, axis=0)

        return X_init, X_holdout, y_init, y_holdout

    def _get_n_samp(self) -> int:
        return round(len(self.x_train) * (1 - self.init_size) * self.acqusition_size)

    @staticmethod
    def _select_acquisition_function(acquisition_function: str = "var") -> Callable:
        log.info(f"Using acquisition function: {acquisition_function}")

        if acquisition_function == "var":
            from acquisition_functions import gp_var

            return gp_var
        elif acquisition_function == "rand":
            from acquisition_functions import gp_rand

            return gp_rand
        elif acquisition_function == "expected_improvement":
            from acquisition_functions import gp_ei

            return gp_ei

    def _select_data_scaler(self, scaler: str = "standard") -> Callable:
        raise NotImplementedError


    def _fill_logs(self):
        raise NotImplementedError

def main():
    # Build Active Learner
    a = ActiveLearner(
        kernel_params=None,
        model_params=None,
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        acquisition_function="var",
        scaler="standard",
        seed=10,
        init_size=0.1,
    )

    a.train(n_iter=20)

    # Build Metrics Handler


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="../datasets/ESOL.csv",
        help="Path to the csv file for the task.",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="ESOL",
        help="str specifying the task. One of [Photoswitch, ESOL, FreeSolv, Lipophilicity].",
    )
    parser.add_argument(
        "-r",
        "--representation",
        type=str,
        default="fingerprints",
        help="str specifying the molecular representation. "
        "One of [SMILES, fingerprints, fragments, fragprints].",
    )
    parser.add_argument(
        "-pca",
        "--use_pca",
        type=bool,
        default=False,
        help="If True apply PCA to perform Principal Components Regression.",
    )
    parser.add_argument(
        "-n",
        "--n_trials",
        type=int,
        default=1,
        help="int specifying number of random train/test splits to use",
    )
    parser.add_argument(
        "-ts",
        "--test_set_size",
        type=float,
        default=0.2,
        help="float in range [0, 1] specifying fraction of dataset to use as test set",
    )
    parser.add_argument(
        "-rms",
        "--use_rmse_conf",
        type=bool,
        default=True,
        help="bool specifying whether to compute the rmse confidence-error curves or the mae "
        "confidence-error curves. True is the option for rmse.",
    )
    parser.add_argument(
        "-pr",
        "--precompute_repr",
        type=bool,
        default=True,
        help="bool indicating whether to precompute representations",
    )
    args = parser.parse_args()

    data_loader = TaskDataLoader(task="ESOL", path="../datasets/ESOL.csv")
    smiles_list, y = data_loader.load_property_data()
    X = featurise_mols(smiles_list, representation="fingerprints")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=10
    )

    a = ActiveLearner(
        kernel_params=None,
        model_params=None,
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        acquisition_function="var",
        scaler="standard",
        seed=10,
        init_size=0.1,
        acquisition_size=0.025
    )

    a.train(n_iter=20)
    """
    kernel_params = {
        "lengthscale": np.ones,
        "noise_variance": None
    }
    
    model_params: {
    
    }
    """

