import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), "..")
if env_path not in sys.path:
    sys.path.append(env_path)


def run_evalutation_on_sequence(gt_label, pred_label):
    """Evaluate the predictions for a single sequence.
    args:
        gt_labels: Numpy array holding the gt labels for a sequence.
        pred_labels: Numpy array holding the predicted labels for a sequence.
    """
    # TODO: Implement this function.


def run_evaluation(gt_labels, pred_labels, output_file):
    """Run the evaluation of the tracker output.
    args:
        gt_labels: Path to ground truth labels (npy).
        pred_labels: Path to predicted labels (npy).
        output_file: Path to output file.
    """
    # TODO: Implement this function.


def main():
    parser = argparse.ArgumentParser(description="Run the evaluation.")
    parser.add_argument("gt_labels", type=str, help="Path to ground truth labels (npy).")
    parser.add_argument("pred_labels", type=str, help="Path to predicted labels (npy).")
    parser.add_argument("output_file", type=str, help="Path to output file.")
    args = parser.parse_args()

    run_evaluation(args.gt_labels, args.pred_labels, args.output_file)


if __name__ == "__main__":
    main()
