import sys, os
from loguru import logger
import typer
from streamlit.web import cli
from streamlit import runtime

from svsvllm.app.settings import TEST_MODE
from svsvllm.app.ui import ui
from svsvllm.utils.logger import set_up_logging

# Define app
app = typer.Typer()


@app.command()
def chatbot() -> None:
    """Main app."""
    logger.trace(f"Creating app...")
    ui()


if __name__ == "__main__":
    set_up_logging()

    # In tests, we run this
    if TEST_MODE:
        chatbot()

    # Main application
    else:
        if runtime.exists():
            logger.trace("Runtime exists")
            app()
        else:
            logger.trace("Creating new runtime")
            cli.main_run([__file__, "--server.port", "8501"])
