import json
import numpy

from models.clf_features import clf_features_runner

# This script is responsible for running experiments.
# flag one_exp = True runs only one experiment
# if False then the runner is called multiple times and the results (val loss, accuracy, f1)
# are saved in a json file

one_exp = False

if one_exp:
    loss, acc, f1 = clf_features_runner("concat.yaml")
else:
    yamls = ["concat.yaml", "affine.yaml"]
    name_metrics = ["loss", "acc", "f1"]
    experiments = {}

    for yaml in yamls:
        experiments[yaml] = {}
        val_losses = []
        accs = []
        f1s = []
        for _ in range(10):
            val_loss, acc, f1 = clf_features_runner(yaml)
            val_losses.append(val_loss)
            accs.append(acc)
            f1s.append(f1)
        metrics = [val_losses, accs, f1s]
        for i, metric in enumerate(metrics):
            mean = numpy.array(metric).mean(axis=0)
            std = numpy.array(metric).std(axis=0)

            experiments[yaml][name_metrics[i]] = (mean, std)
    with open("experiments.json", 'w') as f:
        json.dump(experiments, f)






