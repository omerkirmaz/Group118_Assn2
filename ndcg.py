import numpy as np
import pandas as pd
import statistics
from sklearn.metrics import dcg_score


def ndcg_score(true_filepath, pred_filepath):
    """
    takes the true scores and predicted score and calculates an ndcg score (using sklearn)
    returns: averaged ndcg score over all queries
    """
    true_df = pd.read_csv(true_filepath)
    pred_df = pd.read_csv(pred_filepath)

    all_srch_ids = set(true_df['prop_id'].values)
    # OR:
    # all_srch_ids = true_df['prop-id'].uniques()
    ndcgs = []

    for srch_id in all_srch_ids:
        gold_srch = true_df[true_df['srch_id'] == srch_id].to_list()
        pred_srch = pred_df[pred_df['srch_id'] == srch_id].to_list()

        true_scores = np.asarray([gold_srch['prop_id'].to_list()])
        pred_scores = np.asarray([pred_srch['prop_id'].to_list()])

        dcg = dcg_score(true_scores, pred_scores)
        idcg = dcg_score(true_scores, true_scores)

        ndcg = dcg / idcg
        ndcgs = ndcgs.append(ndcg)

    return statistics.mean(ndcgs)
