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
        self.stop_thread = False
        self.sleep = sleep

    def print_elapsed_time(self, start_time: float) -> None:
        """Show elapsed time."""
        while not self.stop_thread:
            elapsed_time = time.time() - start_time
            print(f"\rElapsed time: {elapsed_time:.2f} seconds", end="")
            time.sleep(self.sleep)

    def run(self, command: ty.Callable, *args: ty.Any, **kwargs: ty.Any) -> ty.Any:
        """Time a command to run."""
        self.stop_thread = False
        start_time = time.time()

        # Start a thread to print the elapsed time
        timer_thread = threading.Thread(target=self.print_elapsed_time, args=(start_time,))
        timer_thread.start()

        # Run the command
        out = command(*args, **kwargs)

        # Stop the timer thread
        self.stop_thread = True
        timer_thread.join()
        elapsed_time = time.time() - start_time
        print(f"\nCommand completed in {elapsed_time:.2f} seconds")
        return out
