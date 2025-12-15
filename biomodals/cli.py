"""Helper script for constructing actual modal run commands."""

import importlib
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

# ruff: noqa: S603
APP_HOME = Path(__file__).parent.resolve() / "app"


app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True, no_args_is_help=True)
def callback():
    """Biomodals CLI - List and get help for biomodals applications.

    This CLI helps users discover available biomodals applications and view their help documentation.
    """


@app.command(name="list")
def list_available_apps(
    use_absolute_paths: Annotated[
        bool,
        typer.Option("--absolute", "-a", help="Use absolute paths for app locations."),
    ] = False,
):
    """Show a list of all available biomodals applications."""
    table = Table("App name", "App path", "Category")
    cwd = Path.cwd()

    for app_file in APP_HOME.glob("*/*_app.py"):
        app_path = (
            app_file.resolve()
            if use_absolute_paths
            else app_file.relative_to(cwd, walk_up=True)
        )
        app_name = app_file.stem.replace("_app", "")
        app_category = app_file.parent.name

        table.add_row(f"[green]{app_name}[/green]", str(app_path), app_category)

    console.print(
        "\n:dna: To see help for an application, use:\n"
        "     [bold]biomodals help <[green]app-name[/green]>[/bold]"
        " or [bold]biomodals help <[green]app-path[/green]>[/bold]"
    )
    console.print(
        "\n:dna: To run an application on [link=https://modal.com]modal.com[/link], use:\n"
        r"     [bold]modal run <[green]app-path[/green]>[/bold] [gray]\[OPTIONS][/gray]"
    )
    console.print("\n:dna: [bold]Available biomodals applications:[/bold]")
    console.print(table)


@app.command(name="help")
def show_app_help(
    app_name: Annotated[
        str, typer.Argument(help="Name or path of the app to show help for.")
    ],
):
    """Show help for a specific biomodals application."""
    app_file = Path(app_name)
    if app_file.exists():
        app_path = app_file
    else:
        app_path = None
        for f in APP_HOME.glob(f"*/{app_name}_app.py"):
            app_path = f
            break
        if app_path is None:
            console.print(
                f"[bold red]Error:[/bold red] Application '{app_name}' not found."
            )
            raise typer.Exit(code=1)

    module_path = (
        str(app_path.relative_to(APP_HOME))
        .replace("/", ".")
        .replace("\\", ".")
        .replace(".py", "")
        .replace("-", "_")
    )
    module_path = f"biomodals.app.{module_path}"
    try:
        module = importlib.import_module(module_path)

        console.print(
            f"[bold]Help for application '[green]{app_path}[/green]':[/bold]\n"
        )
        if docstring := module.__doc__:
            rendered_doc = Markdown(docstring)
            console.print(rendered_doc)
        else:
            console.print("No documentation available.")
    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] Failed to import '{module_path}'")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
