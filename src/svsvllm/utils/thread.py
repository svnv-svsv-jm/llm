__all__ = ["CommandTimer"]

import typing as ty
import time
import threading


class CommandTimer:
    """Run a command and time it."""

    def __init__(self, name: str = "no-process-name", sleep: float = 5) -> None:
        """
        Args:
            name (str, optional):
                The name of the process.

            sleep (float, optional):
                Sleep time in seconds, between two prints.
                Defaults to `5`.
        """
        self.name = name
        self.sleep = sleep
        # Private
        self.stop_thread = False
        self.start_time: float
        self.timer_thread: threading.Thread

    def format_elapsed_time(self, seconds: float) -> str:
        """Format elapsed time."""
        days = seconds // (24 * 3600)
        seconds = seconds % (24 * 3600)
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{int(days):02}:{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    def print_elapsed_time(self, start_time: float) -> None:
        """Show elapsed time."""
        while not self.stop_thread:
            elapsed_time = self.format_elapsed_time(time.time() - start_time)
            print(f"\r[{self.name}] Elapsed time: {elapsed_time} seconds", end="")
            time.sleep(self.sleep)

    def start(self) -> None:
        """Start the timer thread."""
        self.stop_thread = False
        self.start_time = time.time()
        self.timer_thread = threading.Thread(
            target=self.print_elapsed_time,
            args=(self.start_time,),
        )
        self.timer_thread.start()

    def stop(self) -> None:
        """Stop the timer thread."""
        self.stop_thread = True
        self.timer_thread.join()
        elapsed_time = time.time() - self.start_time
        print(f"\nCommand {self.name} completed in {elapsed_time:.2f} seconds")

    def run(self, command: ty.Callable, *args: ty.Any, **kwargs: ty.Any) -> ty.Any:
        """Time a command to run."""
        # Start the timer thread.
        self.start()
        # Run the command
        out = command(*args, **kwargs)
        # Stop the timer thread
        self.stop()
        return out

    def __enter__(self, *args: ty.Any, **kwargs: ty.Any) -> "CommandTimer":
        """Start the timer thread."""
        self.start()
        return self

    def __exit__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        """Stop the timer thread."""
        self.stop()
