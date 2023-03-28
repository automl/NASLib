import numpy as np
from scipy import stats
from sklearn import metrics
import logging

logger = logging.getLogger(__name__)

# Calculate the P@topK, P@bottomK, and Kendall-Tau in predicted topK/bottomK


def p_at_tb_k(predict_scores, true_scores, ks=[1, 5, 10, 20, 25, 30, 50, 75, 100]):
    # ratios=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]):
    predict_scores = np.array(predict_scores)
    true_scores = np.array(true_scores)
    predict_inds = np.argsort(predict_scores)[::-1]
    num_archs = len(predict_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    patks = []
    for k in ks:
        # k = int(num_archs * ratio)
        if k < 1:
            continue
        top_inds = predict_inds[:k]
        bottom_inds = predict_inds[num_archs - k:]
        p_at_topk = len(np.where(true_ranks[top_inds] < k)[0]) / float(k)
        p_at_bottomk = len(
            np.where(true_ranks[bottom_inds] >= num_archs - k)[0]) / float(k)
        kd_at_topk = stats.kendalltau(
            predict_scores[top_inds], true_scores[top_inds]).correlation
        kd_at_bottomk = stats.kendalltau(
            predict_scores[bottom_inds], true_scores[bottom_inds]).correlation
        # [ratio, k, P@topK, P@bottomK, KT in predicted topK, KT in predicted bottomK]
        patks.append((k / len(true_scores), k, p_at_topk,
                     p_at_bottomk, kd_at_topk, kd_at_bottomk))
    return patks


# Calculate the BR@K, WR@K
def minmax_n_at_k(predict_scores, true_scores, ks=[1, 5, 10, 20, 25, 30, 50, 75, 100]):
    true_scores = np.array(true_scores)
    predict_scores = np.array(predict_scores)
    num_archs = len(true_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    predict_best_inds = np.argsort(predict_scores)[::-1]
    minn_at_ks = []
    for k in ks:
        ranks = true_ranks[predict_best_inds[:k]]
        if len(ranks) < 1:
            continue
        minn = int(np.min(ranks)) + 1
        maxn = int(np.max(ranks)) + 1
        minn_at_ks.append((k, k, minn, float(minn) / num_archs,
                          maxn, float(maxn) / num_archs))
    return minn_at_ks


def compute_scores(ytest, test_pred):
    ytest = np.array(ytest)
    test_pred = np.array(test_pred)
    METRICS = [
        "mae",
        "rmse",
        "pearson",
        "spearman",
        "kendalltau",
        "kt_2dec",
        "kt_1dec",
        "full_ytest",
        "full_testpred",
    ]
    metrics_dict = {}

    try:
        precision_k_metrics = p_at_tb_k(test_pred, ytest)

        for metric in precision_k_metrics:
            k, p_at_topk, kd_at_topk = metric[1], metric[2], metric[4]
            metrics_dict[f'p_at_top{k}'] = p_at_topk
            metrics_dict[f'kd_at_top{k}'] = kd_at_topk

        best_k_metrics = minmax_n_at_k(test_pred, ytest)

        for metric in best_k_metrics:
            k, min_at_k = metric[1], metric[3]
            metrics_dict[f'br_at_{k}'] = min_at_k

        metrics_dict["mae"] = np.mean(abs(test_pred - ytest))
        metrics_dict["rmse"] = metrics.mean_squared_error(
            ytest, test_pred, squared=False
        )
        metrics_dict["pearson"] = np.abs(np.corrcoef(ytest, test_pred)[1, 0])
        metrics_dict["spearman"] = stats.spearmanr(ytest, test_pred)[0]
        metrics_dict["kendalltau"] = stats.kendalltau(ytest, test_pred)[0]
        metrics_dict["kt_2dec"] = stats.kendalltau(
            ytest, np.round(test_pred, decimals=2)
        )[0]
        metrics_dict["kt_1dec"] = stats.kendalltau(
            ytest, np.round(test_pred, decimals=1)
        )[0]
        for k in [10, 20]:
            top_ytest = np.array(
                [y > sorted(ytest)[max(-len(ytest), -k - 1)] for y in ytest]
            )
            top_test_pred = np.array(
                [
                    y > sorted(test_pred)[max(-len(test_pred), -k - 1)]
                    for y in test_pred
                ]
            )
            metrics_dict["precision_{}".format(k)] = (
                sum(top_ytest & top_test_pred) / k
            )
        metrics_dict["full_ytest"] = ytest.tolist()
        metrics_dict["full_testpred"] = test_pred.tolist()

    except:
        for metric in METRICS:
            metrics_dict[metric] = float("nan")
    if np.isnan(metrics_dict["pearson"]) or not np.isfinite(
            metrics_dict["pearson"]
    ):
        logger.info("Error when computing metrics. ytest and test_pred are:")
        logger.info(ytest)
        logger.info(test_pred)

    return metrics_dict
