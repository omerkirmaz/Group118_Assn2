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

    all_srch_ids = set(true_df['srch_id'].to_list())

    ndcgs = []

    for srch_id in all_srch_ids:

        gold_srch = true_df[true_df['srch_id'] == srch_id]
        gold_list = gold_srch['prop_id'].to_list()

        pred_srch = pred_df[pred_df['srch_id'] == srch_id]
        pred_list = pred_srch['prop_id'].to_list()

        true_scores = np.asarray([gold_list])
        pred_scores = np.asarray([pred_list])

        dcg = dcg_score(true_scores, pred_scores, k=5)
        idcg = dcg_score(true_scores, true_scores, k=5)

        ndcg = dcg / idcg
        ndcgs.append(ndcg)

    return statistics.mean(ndcgs)
