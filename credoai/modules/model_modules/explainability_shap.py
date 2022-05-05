from absl import logging
from pydantic import NonNegativeFloat
from credoai.utils.common import to_array, NotRunError, ValidationError
from credoai.modules.credo_module import CredoModule

from typing import List, Union
import pandas as pd
import shap

class SHAPModule(CredoModule):
    """
    SHAP explainability module for Credo AI. 

    Parameters
    ----------
    model : 
    data : 
    """
    def __init__(self,
                 model,
                 data):
        self.model = model
        self.data = data
        self.explainer = None
        self.shap_values = None

    def run(self):
        """
        Run shap explainability module
         
        Returns
        -------
        dict
            Dictionary containing one pandas Dataframes:
                - "disaggregated results": The disaggregated performance metrics, along with acceptability and risk
            as columns
        """
        self.explainer, self.shap_values = self._setup_explainer()
        feature_names = shap_values.feature_names
        shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
        self.results = {'shap_values': shap_df,
                        'mean_abs_shap': shap_df.abs().mean().sort_values(ascending=False)}
        return self
        
    def prepare_results(self, filter=None):
        return 

    def explain_observation(self, idx):
        """Create a waterfall plot for a particular observation
        
        Parameters
        ----------
        idx : int
            row index for data passed to SHAPModule
        """
        shap.waterfall_plot(self.shap_values[idx])

    def _get_explainer(self):
        explainer = shap.KernelExplainer(self.model)
        shap_values = explainer(self.data)
        return explainer, shap_values