import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format= "[%(asctime)s: %(levelname)s]: %(message)s"
    )

# while True:
#     project_name = input("Enter the Project Name: ")
#     if project_name != '':
#         break
project_name = "us_visa"

# logging.info(f"Creating project by name: {project_name}")

# list of files:
list_of_files = [
    # ".github/workflows/.gitkeep",
    f"{project_name}/__init__.py",
    f"{project_name}/cloud_storage/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/constant/__init__.py",
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/data_access/__init__.py",
    # "configs/config.yaml",
    # "init_setup.sh",
    # "requirements.txt",
    "setup.py",
    # "pyproject.toml",
    # 'setup.cfg',
    # 'README.md',
    # "Dockerfile"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating a directory at: {filedir} for file: {filename}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating a new file: {filename} at path: {filepath}")
    else:
        logging.info(f"file is already present at: {filepath}")