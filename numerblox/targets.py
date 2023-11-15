import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from typing import List, Union
from abc import abstractmethod
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import BayesianGaussianMixture
from sklearn.utils.validation import check_is_fitted

# Ignore SettingWithCopyWarning
pd.options.mode.chained_assignment = None

class BaseTargetProcessor(BaseEstimator, TransformerMixin):
    """Common functionality for preprocessors and postprocessors."""

    def __init__(self):
        ...

    def fit(self, X, y=None):
        return self

    @abstractmethod
    def transform(
        self, X: Union[np.array, pd.DataFrame], y=None, **kwargs
    ) -> pd.DataFrame:
        ...

    def __call__(
        self, X: Union[np.array, pd.DataFrame], y=None, **kwargs
    ) -> pd.DataFrame:
        return self.transform(X=X, y=y, **kwargs)
    
    @abstractmethod
    def get_feature_names_out(self, input_features=None) -> List[str]:
        ...


class BayesianGMMTargetProcessor(BaseTargetProcessor):
    """
    Generate synthetic (fake) target using a Bayesian Gaussian Mixture model. \n
    Based on Michael Oliver's GitHub Gist implementation: \n
    https://gist.github.com/the-moliver/dcdd2862dc2c78dda600f1b449071c93

    :param n_components: Number of components for fitting Bayesian Gaussian Mixture Model.
    """
    def __init__(
        self,
        n_components: int = 3,
    ):
        super().__init__()
        self.n_components = n_components
        self.ridge = Ridge(fit_intercept=False)
        self.bins = [0, 0.05, 0.25, 0.75, 0.95, 1]

    def fit(self, X: pd.DataFrame, y: pd.Series, eras: pd.Series):
        """
        Fit Bayesian Gaussian Mixture model on coefficients and normalize.
        :param X: DataFrame containing features.
        :param y: Series containing real target.
        :param eras: Series containing era information.
        """
        bgmm = BayesianGaussianMixture(n_components=self.n_components)
        coefs = self._get_coefs(dataf=X, y=y, eras=eras)
        bgmm.fit(coefs)
        # make probability of sampling each component equal to better balance rare regimes
        bgmm.weights_[:] = 1 / self.n_components
        self.bgmm_ = bgmm
        return self

    def transform(self, X: pd.DataFrame, eras: pd.Series) -> np.array:
        """
        Main method for generating fake target.
        :param X: DataFrame containing features.
        :param eras: Series containing era information.
        """
        check_is_fitted(self, "bgmm_")
        assert len(X) == len(eras), "X and eras must be same length."
        all_eras = eras.unique().tolist()
        # Scale data between 0 and 1
        X = X.astype(float)
        X /= X.max()
        X -= 0.5
        X.loc[:, 'era'] = eras

        fake_target = self._generate_target(dataf=X, all_eras=all_eras)
        return fake_target

    def _get_coefs(self, dataf: pd.DataFrame, y: pd.Series, eras: pd.Series) -> np.ndarray:
        """
        Generate coefficients for BGMM.
        :param dataf: DataFrame containing features.
        :param y: Series containing real target.
        """
        coefs = []
        dataf.loc[:, 'era'] = eras
        dataf.loc[:, 'target'] = y
        all_eras = dataf['era'].unique().tolist()
        for era in all_eras:
            era_df = dataf[dataf['era'] == era]
            era_y = era_df.loc[:, 'target']
            era_df = era_df.drop(columns=["era", "target"])
            self.ridge.fit(era_df, era_y)
            coefs.append(self.ridge.coef_)
        stacked_coefs = np.vstack(coefs)
        return stacked_coefs

    def _generate_target(
        self, dataf: pd.DataFrame, all_eras: list
    ) -> np.ndarray:
        """Generate fake target using Bayesian Gaussian Mixture model."""
        fake_target = []
        for era in tqdm(all_eras, desc="Generating fake target"):
            features = dataf[dataf['era'] == era]
            features = features.drop(columns=["era", "target"])
            # Sample a set of weights from GMM
            beta, _ = self.bgmm_.sample(1)
            # Create fake continuous target
            fake_targ = features @ beta[0]
            # Bin fake target like real target
            fake_targ = (rankdata(fake_targ) - 0.5) / len(fake_targ)
            fake_targ = (np.digitize(fake_targ, self.bins) - 1) / 4
            fake_target.append(fake_targ)
        return np.concatenate(fake_target)
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return feature names."""
        return ["fake_target"] if not input_features else input_features
    

class SignalsTargetProcessor(BaseTargetProcessor):
    """
    Engineer targets for Numerai Signals. \n
    More information on implements Numerai Signals targets: \n
    https://forum.numer.ai/t/decoding-the-signals-target/2501

    :param price_col: Column from which target will be derived. \n
    :param windows: Timeframes to use for engineering targets. 10 and 20-day by default. \n
    :param bins: Binning used to create group targets. Nomi binning by default. \n
    :param labels: Scaling for binned target. Must be same length as resulting bins (bins-1). Numerai labels by default.
    """

    def __init__(
        self,
        price_col: str = "close",
        windows: list = None,
        bins: list = None,
        labels: list = None,
    ):
        super().__init__()
        self.price_col = price_col
        self.windows = windows if windows else [10, 20]
        self.bins = bins if bins else [0, 0.05, 0.25, 0.75, 0.95, 1]
        self.labels = labels if labels else [0, 0.25, 0.50, 0.75, 1]

    def transform(self, dataf: pd.DataFrame, eras: pd.Series) -> np.array:
        for window in tqdm(self.windows, desc="Signals target engineering windows"):
            dataf.loc[:, f"target_{window}d_raw"] = (
                dataf[self.price_col].pct_change(periods=window).shift(-window)
            )
            era_groups = dataf.groupby(eras)

            dataf.loc[:, f"target_{window}d_rank"] = era_groups[
                f"target_{window}d_raw"
            ].rank(pct=True, method="first")
            dataf.loc[:, f"target_{window}d_group"] = era_groups[
                f"target_{window}d_rank"
            ].transform(
                lambda group: pd.cut(
                    group, bins=self.bins, labels=self.labels, include_lowest=True
                )
            )
        output_cols = self.get_feature_names_out()
        return dataf[output_cols].to_numpy()

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return feature names of Signals targets. """
        if not input_features:
            feature_names = []
            for window in self.windows:
                feature_names.append(f"target_{window}d_raw")
                feature_names.append(f"target_{window}d_rank")
                feature_names.append(f"target_{window}d_group")
        else:
            feature_names = input_features
        return feature_names
