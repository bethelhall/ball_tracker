import json
import argparse
import os
import glob
import tabulate


def get_parser():
    parser = argparse.ArgumentParser(
        description="Inspect metrics.json files(output of detectron2 training)")
    parser.add_argument("--folder",
                        default="../data/ball_output/COCO-Detection",
                        help="folder where training outputs reside",
                        )
    return parser


def get_training_metrics_files(dir):
    return sorted(glob.glob(f"{dir}/*/metrics.json"))


def getMetrics(file_name, keys):
    with open(file_name) as json_file:
        last_line = list(json_file)[-1]
        data = json.loads(last_line)
        model_name = get_metric_folder_name(file_name)
        row = [model_name, *[data[key] for key in keys]]

        return row

def get_metric_folder_name(file_name):
    metric_folder, _ = os.path.split(file_name)
    _, model_folder_name = os.path.split(metric_folder)
    return model_folder_name

if __name__ == "__main__":
    args = get_parser().parse_args()

    metrics_files = get_training_metrics_files(args.folder)

    headers = ['loss_cls', 'loss_box_reg', 'total_loss', 'iteration']
    rows = []
    for file_name in metrics_files:
        rows.append(getMetrics(file_name, headers))

    print(tabulate.tabulate(rows, headers=headers, tablefmt='orgtbl'))