import numpy as np
import textwrap

from credoai.utils.common import ValidationError
from scipy import stats
from sklearn.utils import check_consistent_length
class Sampler():
    def __init__(self, verbose=True):
        self.verbose=verbose
    
    def _pooled_sd(self, arr1, arr2):
        n1 = len(arr1)
        n2 = len(arr2)
        sd1 = np.std(arr1)
        sd2 = np.std(arr2)
        return (((n1 - 1)*sd1**2 + (n2-1)*sd2**2) / (n1 + n2-2))**0.5

    def _cohens_d(self, arr1, arr2):
        """Calculates Cohens D"""
        if self.verbose:
            print(f'Mean Statistic 1: {np.mean(arr1)}', 
                f'Mean Statistic 2: {np.mean(arr2)}')
        return (np.mean(arr1)-np.mean(arr2)) / self._pooled_sd(arr1, arr2)


    def _get_sample_indices(self, arr_length, n_rep=1000, sample_size=None):
        """Get indices for bootstrap

        Parameters
        ----------
        arr_length : int
            length of array to sample
        n_rep : int, optional
            number of times to sample, by default 1000
        sample_size : int, optional
            length of each sample. If none, the sample
            will be the same arr_length, as in the classic
            bootstrap, by default None

        Returns
        -------
        np.array
            array of samples where each column is a different sample.
            Size: sample_size X n_rep
        """        
        if sample_size is None:
            sample_size = arr_length
        return np.random.choice(np.arange(arr_length), size=[sample_size,n_rep], replace=True)

    def _apply_fun(self, samples, fun):
        """Applies function to samples

        Samples must have the keys y_true and y_pred

        Parameters
        ----------
        samples : dict
            output of create_samples
        fun : callable
            function to apply to each sample

        Returns
        -------
        np.array
            array with length of n_rep (number of samples) with function applied
        """        
        try:
            zipped = zip(samples['y_true'].T, 
                         samples['y_pred'].T)
            return np.array([fun(true, pred) for true, pred in zipped])
        except:
            return np.array([fun(s) for s in samples['y_pred'].T])

    def create_samples(self, array_dict, n_rep=1000, sample_size=None):
        """Samples a dictionary of arrays

        Parameters
        ----------
        array_dict : dict
            Dictionary of arrays with arbitrary labeling keys. Each array
            must be the same length. Each sample will take the same index
            from each array (e.g., )
        n_rep : int, optional
            number of times to sample, by default 1000
        sample_size : int, optional
            length of each sample. If none, the sample
            will be the same arr_length, as in the classic
            bootstrap, by default None

        Returns
        -------
        dict
            Dictionary with same keys as array_dict, but with matrices of
            resampled arrays. Size: sample_size X n_rep

        Raises
        ------
        ValidationError
            Error raised if arrays are not the same length
        """        
        try:
            check_consistent_length(*array_dict.values())
        except ValueError:
            raise ValidationError("All arrays passed to bootstrap must have the same length")

        array_length = len(array_dict[list(array_dict.keys())[0]])
        indices = self._get_sample_indices(array_length, n_rep=n_rep, sample_size=sample_size)
        out = {k: v[indices] for k, v in array_dict.items()}
        return out


    def evaluate(self, statistics1, statistics2, thresh=.01):
        """Evaluates whether two statistics are statistically different

        Uses Cohens D to calculate effect size

        Parameters
        ----------
        statistics1 : np.array
            the first distribution of statistics
        statistics2 : np.array
            the second distribution of statistics
        thresh : float, optional
            statistical threshold, below which the null 
            (that the statistics are the same) is rejected, by default .01

        Returns
        -------
        [type]
            [description]
        """        
        # compare distributions of r2 scores
        effect_size = self._cohens_d(statistics1, statistics2)
        p_val = stats.norm.sf(abs(effect_size))*2
        rejected = False if p_val > thresh else True # good if not statistically different

        return  {'effect_size': effect_size,
                 'p_val': p_val,
                 'rejected': rejected} 

    # function to evaluate 
    def evaluate_preds(self, pred1, true1, pred2, true2, 
                 eval_fun, sample_size, n_boot=2000):
        samples1 = self.create_samples({'y_pred': pred1, 'y_true': true1},
                                  n_boot, sample_size=sample_size)
        samples2 = self.create_samples({'y_pred': pred2, 'y_true': true2},
                                  n_boot, sample_size=sample_size)
        # on each sample compute r2_score
        statistics1 = self._apply_fun(samples1, eval_fun)
        statistics2 = self._apply_fun(samples2, eval_fun)
        return self.evaluate(statistics1, statistics2)


class RiskQuestionnare():
    def __init__(self, 
            big_deal=.01,
            medium_deal = .01,
            small_deal = .01
        ):
        self.deals = {
            'big': big_deal,
            'medium': medium_deal,
            'small': small_deal
        }

    def _find_n(self, p1, p2, test_statistic):
        p_combined = (p1+p2)/2
        num = p_combined*(1-p_combined)*2
        denom = ((p1-p2)/test_statistic)**2
        return num/denom

    def run(self):
        base = .8
        output = {}
        for deal, important_p_val in self.deals.items():
            sig_val = stats.norm.ppf(1-important_p_val/2) # zval equal to pval of .05

            print("The priveleged group has an accuracy of 80%...\n")
            question_text = textwrap.fill("How much worse would "
                "the accuracy for the unpriveleged group have to be to be a "
                f"'{deal.upper()} deal'", 50)
            significant_accuracy_diff = float(input(question_text))/100

            n_up = self._find_n(base+significant_accuracy_diff, base, sig_val)
            n_down = self._find_n(base-significant_accuracy_diff, base, sig_val)
            average_n = (n_up+n_down)/2
            print(f'Number of people to use for this test: [{n_down}, {n_up}]\nAverage: {average_n}')
            sample_size = int(average_n)
            output[deal] = {'sample_size': sample_size, 'thresh': important_p_val}
        return output