import sys
import os
import glob
import numpy as np

from ogb.lsc import MAG240MEvaluator
import torch


input_path, output_path = sys.argv[1:3]
filenames = glob.glob(os.path.join(input_path, "seed*", "cv-*", "y_pred_test.npy"))

y_pred_test_all = []
for fname in filenames:
    print(fname)
    y = torch.as_tensor(np.load(fname)).softmax(axis=1).numpy()
    y_pred_test_all.append(y)
nsample, ndim = y_pred_test_all[0].shape

y_pred_test = np.concatenate(y_pred_test_all).reshape((-1, nsample, ndim)).mean(axis=0)
res = {'y_pred': y_pred_test.argmax(axis=1)}

evaluator = MAG240MEvaluator()
print("Saving SUBMISSION to %s" % output_path)
evaluator.save_test_submission(res, output_path)
