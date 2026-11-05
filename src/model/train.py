#from clearml import Task
from ultralytics import YOLO
import os
import argparse
import yaml

DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))


def make_absolute_yaml(dataset):
    """Writes <dataset>_train.yaml with an absolute 'path' so Ultralytics finds the images
    regardless of the working directory. Returns the yaml path."""
    dataset_dir = os.path.join(DATASETS_DIR, dataset)
    src = os.path.join(dataset_dir, f"{dataset}.yaml")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"No dataset yaml at {src}. Generate the dataset first.")
    with open(src) as f:
        data = yaml.safe_load(f)
    data["path"] = dataset_dir
    out = os.path.join(dataset_dir, f"{dataset}_train.yaml")
    with open(out, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return out


def train():
    '''Fine-tunes a YOLO model on a generated dataset (default: start from models/baseline.pt)'''

    model = YOLO(args.model)
    data_yaml = make_absolute_yaml(args.dataset)
    print(f"Training on {data_yaml} with weights {args.model}")

    model.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, name=args.name)
    model.val()
    model.export()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset", default="sample_small", help="Dataset name (folder inside 'datasets') to train on", type=str)
    parser.add_argument("-m", "--model", dest="model", default=os.path.join(os.path.dirname(__file__), "..", "..", "models", "baseline.pt"),
                        help="Starting weights (.pt) for fine-tuning, or a model yaml (e.g. yolov8s.yaml) to train from scratch", type=str)
    parser.add_argument("-e", "--epochs", dest="epochs", default=100, help="Training epochs", type=int)
    parser.add_argument("--imgsz", dest="imgsz", default=1280, help="Training image size. Use the camera width (1280) so distant aircraft stay visible", type=int)
    parser.add_argument("-b", "--batch", dest="batch", default=-1, help="Batch size (-1 = auto)", type=int)
    parser.add_argument("--device", dest="device", default=None, help="Device, e.g. 0 for first GPU or cpu", type=str)
    parser.add_argument("--name", dest="name", default=None, help="Run name under runs/detect", type=str)
    global args
    args = parser.parse_args()
    train()
