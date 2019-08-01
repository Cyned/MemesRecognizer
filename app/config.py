import os

# path to the application
APP_DIR: str = str(os.path.dirname(os.path.abspath(__file__)))

# path to project
PROJECT_DIR: str = APP_DIR[:APP_DIR.rfind('/')]

DATA_DIR = os.path.join(PROJECT_DIR, 'data/')

GPU: bool = False
