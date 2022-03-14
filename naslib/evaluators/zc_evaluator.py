import codecs
import time
import json
import logging
import os
import numpy as np
import torch
from scipy import stats
from sklearn import metrics

from naslib.search_spaces.core.query_metrics import Metric

logger = logging.getLogger(__name__)


class PredictorEvaluator(object):
    """
    This class will evaluate a chosen predictor based on
    correlation and rank correlation metrics, for the given
    initialization times and query times.
    """

    def __init__(self, predictor, config=None):
        self.predictor = predictor
        self.config = config
        self.test_size = config.test_size
        self.train_size = config.train_size
        self.dataset = config.dataset
        self.metric = Metric.VAL_ACCURACY
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results = [config]

        self.train_data_file = config.train_data_file
        self.test_data_file = config.test_data_file

    def adapt_search_space(
        self, search_space, load_labeled, scope=None, dataset_api=None
    ):
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.predictor.set_ss_type(self.search_space.get_type())
        self.load_labeled = load_labeled
        self.dataset_api = dataset_api

    def get_full_arch_info(self, arch):
        """
        Given an arch, return the accuracy, train_time,
        and also a dict of extra info if required by the predictor
        """
        info_dict = {}
        accuracy = arch.query(
            metric=self.metric, dataset=self.dataset,
            dataset_api=self.dataset_api
        )
        train_time = arch.query(
            metric=Metric.TRAIN_TIME, dataset=self.dataset,
            dataset_api=self.dataset_api
        )
        return accuracy, train_time, info_dict

    def load_dataset_from_file(self, datapath, size):
        with open(datapath) as f:
            data = json.load(f)

        xdata = []
        ydata = []

        for i, x in enumerate(data):
            arch = x['arch']
            acc = x['accuracy']
            model = self.search_space.clone()
            model.set_spec(arch)
            model.prepare_evaluation()
            model.parse()

            xdata.append(model)
            ydata.append(acc)

            if i >= size:
                break

        return [xdata, ydata, None, None]


    def load_dataset(self, load_labeled=False, data_size=10):
        """
        There are two ways to load an architecture.
        load_labeled=False: sample a random architecture from the search space.
        This works on NAS benchmarks where we can query any architecture (nasbench101/201/301)
        load_labeled=True: sample a random architecture from a set of evaluated architectures.
        When we only have data on a subset of the search space (e.g., the set of 5k DARTS
        architectures that have the full training info).

        After we load an architecture, query the final val accuracy.
        If the predictor requires extra info such as partial learning curve info, query that too.
        """
        xdata = []
        ydata = []
        info = []
        train_times = []
        while len(xdata) < data_size:
            if not load_labeled:
                arch = self.search_space.clone()
                arch.sample_random_architecture(dataset_api=self.dataset_api)
            else:
                arch = self.search_space.clone()
                arch.load_labeled_architecture(dataset_api=self.dataset_api)

            arch.prepare_evaluation()
            arch.parse()

            accuracy, train_time, info_dict = self.get_full_arch_info(arch)
            xdata.append(arch)
            ydata.append(accuracy)
            info.append(info_dict)
            train_times.append(train_time)

        return [xdata, ydata, info, train_times]

    def single_evaluate(self, train_data, test_data):
        """
        Evaluate the predictor.
        """
        xtrain, ytrain, train_info, train_times = train_data
        xtest, ytest, test_info, _ = test_data
        train_size = len(xtrain)

        data_reqs = self.predictor.get_data_reqs()

        logger.info("Fit the predictor")
        fit_time_start = time.time()
        self.predictor.fit(xtrain, ytrain, train_info)
        fit_time_end = time.time()

        test_pred = []

        for graph in xtest:
            pred = self.predictor.query(graph, test_info)
            test_pred.append(pred)
        test_pred = np.array(test_pred)

        query_time_end = time.time()

        # If the predictor is an ensemble, take the mean
        if len(test_pred.shape) > 1:
            test_pred = np.mean(test_pred, axis=0)

        logger.info("Compute evaluation metrics")
        results_dict = self.compare(ytest, test_pred)
        results_dict["train_time"] = -1 if train_times is None else np.sum(train_times)
        results_dict["fit_time"] = fit_time_end - fit_time_start
        results_dict["query_time"] = (query_time_end - fit_time_end) / len(xtest)

        method_type = self.predictor.method_type
        print(
            "dataset: {}, predictor: {}, spearman {}".format(
                self.dataset, method_type, np.round(results_dict["spearman"], 4)
            )
        )
        print("full ytest", results_dict["full_ytest"])
        print("full testpred", results_dict["full_testpred"])

        # print entire results dict:
        print_string = ""
        for key in results_dict:
            if type(results_dict[key]) not in [str, set, bool]:
                # todo: serialize other types
                print_string += key + ": {}, ".format(np.round(results_dict[key], 4))
        logger.info(print_string)
        self.results.append(results_dict)


    def load_train_test_data(self):
        logger.info("Loading the test set")

        if self.test_data_file is not None:
            print('Loading from file')
            test_data = self.load_dataset_from_file(self.test_data_file, self.test_size)
        else:
            test_data = self.load_dataset(
                load_labeled=self.load_labeled, data_size=self.test_size
            )

        logger.info("Loading the training set")

        if self.train_data_file is not None:
            print('Loading from file')
            train_data = self.load_dataset_from_file(self.train_data_file, self.train_size)
        else:
            train_data = self.load_dataset(
                load_labeled=self.load_labeled,
                data_size=self.train_size
            )

        return train_data, test_data

    def evaluate(self):
        self.predictor.pre_process()
        train_data, test_data = self.load_train_test_data()
        self.single_evaluate(train_data, test_data)
        self._log_to_json()

        return self.results

    def compare(self, ytest, test_pred):
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
            "precision_10",
            "precision_20",
            "full_ytest",
            "full_testpred",
        ]
        metrics_dict = {}

        try:
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

    def _log_to_json(self):
        """log statistics to json file"""
        if not os.path.exists(self.config.save):
            os.makedirs(self.config.save)
        with codecs.open(
            os.path.join(self.config.save, "errors.json"), "w", encoding="utf-8"
        ) as file:
            for res in self.results:
                for key, value in res.items():
                    if type(value) == np.int32 or type(value) == np.int64:
                        res[key] = int(value)
                    if type(value) == np.float32 or type(value) == np.float64:
                        res[key] = float(value)

            json.dump(self.results, file, separators=(",", ":"))
