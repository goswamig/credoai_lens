from absl import logging
from pydantic import NonNegativeFloat
from credoai.utils.common import to_array, NotRunError, ValidationError
from credoai.modules.credo_module import CredoModule

from typing import List, Union
import pandas as pd
import shap

class ShapExplainer(CredoModule):
    """
    SHAP explainability module for Credo AI. 

    Parameters
    ----------
    model : 
    data : 
    """
    def __init__(self,
                 model,
                 X):
        self.model = model
        self.X = X
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
        feature_names = self.shap_values.feature_names
        shap_df = pd.DataFrame(self.shap_values.values, columns=feature_names)
        self.results = {'shap_values': shap_df,
                        'mean_abs_shap': shap_df.abs().mean().sort_values(ascending=False)}
        return self
        
    def prepare_results(self, filter=None):
        return 

    def _setup_explainer(self):
        explainer = shap.Explainer(self.model)
        explainer.expected_value = explainer.expected_value[0]
        shap_values = explainer(self.X)
        return explainer, shap_values