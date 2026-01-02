import shutil, os

def save_config(config_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.yaml"
    shutil.copy(config_path, path)

def setup_path(path):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )
