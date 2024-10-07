import typer
from streamlit.web import cli
from streamlit import runtime

from svsvllm.app.ui import ui

# Define app
app = typer.Typer()


@app.command()
def chatbot() -> None:
    """Main app."""
    ui()


if __name__ == "__main__":
    if runtime.exists():
        app()
    else:
        cli.main_run([__file__, "--server.port", "8501"])
