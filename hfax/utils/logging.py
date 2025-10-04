import time
import cProfile, pstats, io
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table

import logging


def setup_config():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler()],
    )


setup_config()
logger = logging.getLogger("rich")
console = Console()


