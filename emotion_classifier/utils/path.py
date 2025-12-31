import shutil, os, pathlib

def save_config(config_path, output_dir):
    path = pathlib.Path(output_dir) / "config.yaml"
    shutil.copy(config_path, path)

def setup_path(path):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )
