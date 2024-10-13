import sys
from loguru import logger
import typer
from streamlit.web import cli
from streamlit import runtime

from svsvllm.app.ui import ui
from svsvllm.utils.logger import set_up_logging

# Define app
app = typer.Typer()


@app.command()
def chatbot() -> None:
    """Main app."""
    ui()


if __name__ == "__main__":
    set_up_logging()
    if runtime.exists():
        app()
    else:
        cli.main_run([__file__, "--server.port", "8501"])
