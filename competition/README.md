## Sample submission

Your predictor class must be in a file named `predictor.py` with the name `ZeroCostPredictor`. You may add more files to support your ZeroCostPredictor (`compute.py` is provided as an example in this case).

See `sample_submission.zip` for an example of a submission that you must upload to CodaLab.

## Evaluation
`evaluate_predictor.py` and `data` are provided here as a rough example of the code which shall be run to evaluate your submission. These should not be a part of your submission.

To evaluate the `ZeroCostPredictor` in `predictor.py` for NASBench201 search space for the task of classifying CIFAR10, for example, run the following command:

```python evaluate_predictor.py --datapath data/nasbench201/cifar10```