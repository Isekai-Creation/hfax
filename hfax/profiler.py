import time
import cProfile
import pstats
import io
import logging
from rich.table import Table
from rich.console import Console

# This assumes that the logger is configured by hfax.utils.logging
logger = logging.getLogger("rich")

class Profiler:
    def __init__(self, name, sort_stats="cumtime", limit=50):
        self.name = name
        self.profiler = cProfile.Profile()
        self.limit = limit
        self.sort_stats = sort_stats

    def __enter__(self):
        self.start_time = time.time()
        self.profiler.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.disable()
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        s = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=s).sort_stats(self.sort_stats)
        stats.print_stats(self.limit)
        profile_text = s.getvalue()

        self.render_stats_table(profile_text)

        logger.info(f"{self.name} took {duration:.4f} seconds\n")

    def render_stats_table(self, profile_text):
        table = Table(title=f"Profile: {self.name}", show_lines=True)
        table.add_column("ncalls", style="cyan", justify="right")
        table.add_column("tottime", style="magenta", justify="right")
        table.add_column("percall", style="magenta", justify="right")
        table.add_column("cumtime", style="green", justify="right")
        table.add_column("percall", style="green", justify="right")
        table.add_column("filename:lineno(function)", style="yellow")

        for line in profile_text.splitlines():
            if line.strip().startswith("ncalls") or line.strip().startswith("Ordered"):
                continue
            if len(line.strip()) < 40:
                continue
            parts = line.strip().split(None, 5)
            if len(parts) == 6:
                table.add_row(*parts)

        Console().print(table)
