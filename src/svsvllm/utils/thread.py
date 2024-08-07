__all__ = ["CommandTimer"]

import typing as ty
import time
import threading


class CommandTimer:
    """Run a command and time it."""

    def __init__(self, sleep: float = 5) -> None:
        """
        Args:
            sleep (float, optional):
                Sleep time in seconds, between two prints.
                Defaults to `5`.
        """
        self.sleep = sleep
        # Private
        self.stop_thread = False
        self.start_time: float
        self.timer_thread: threading.Thread

    def print_elapsed_time(self, start_time: float) -> None:
        """Show elapsed time."""
        while not self.stop_thread:
            elapsed_time = time.time() - start_time
            print(f"\rElapsed time: {elapsed_time:.2f} seconds", end="")
            time.sleep(self.sleep)

    def start(self) -> None:
        """Start the timer thread."""
        self.stop_thread = False
        self.start_time = time.time()
        # Start a thread to print the elapsed time
        self.timer_thread = threading.Thread(
            target=self.print_elapsed_time, args=(self.start_time,)
        )
        self.timer_thread.start()

    def stop(self) -> None:
        """Stop the timer thread."""
        self.stop_thread = True
        self.timer_thread.join()
        elapsed_time = time.time() - self.start_time
        print(f"\nCommand completed in {elapsed_time:.2f} seconds")

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
