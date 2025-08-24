import argparse
import yaml

def load_config(config_path):
    """
    Loads configuration from a YAML file .

    Returns:
        dict: Configuration loaded from the YAML file.
    """
    try:
        with open(config_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML file: {exc}")
                raise
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        raise
    except Exception as e:
        print(f"Unexpected error reading config file: {e}")
        raise

    return config

def load_args():
    """
    Loads command line arguments.
    
    Returns: 
        args: parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Anomaly Detection")
    parser.add_argument("split", nargs="?", choices=["train", "test"])
    parser.add_argument("--config", type=str, required=True, help="path to the yaml config file")
    # required training super-parameters
    parser.add_argument("--checkpoint", type=str, default=None, help="student checkpoint")
    parser.add_argument("--category", type=str , default=None, help="category name for MvTec AD dataset")
    parser.add_argument("--epochs", type=int, default=None, help='number of epochs')

    parser.add_argument("--checkpoint-epoch", type=int, default=None, help="checkpoint resumed for testing (1-based)")
    parser.add_argument("--batch-size", type=int, default=None, help='batch size')
    # trivial parameters
    parser.add_argument("--result-path", type=str, default=None, help="save results")
    parser.add_argument("--save-fig", action='store_true', help="save images with anomaly score")
    parser.add_argument("--mvtec-ad", type=str, default=None, help="MvTec-AD dataset path")
    parser.add_argument('--model-save-path', type=str, default=None, help='path where student models are saved')

    args = parser.parse_args()
    return args