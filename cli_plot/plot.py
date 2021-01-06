#!/usr/bin/env python

"""
TODO
"""

import sys
from typing import List, Tuple, Union, Optional, Any
from pathlib import Path
from enum import Enum
from collections import namedtuple
from itertools import count

import typer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.backend_bases import KeyEvent
import seaborn as sns

from .config import Config
from .mouse_event_handlers import MouseEventHandlers
from .functions import demo_df

__version__ = "0.0.1"


class PlotType(str, Enum):
    """The type of plot to plot. """

    line = "line"
    scatter = "scatter"


class Context(str, Enum):
    """Seaborn context set the size of the plot  """

    # Smallest to largest
    paper = "paper"
    notebook = "notebook"
    talk = "talk"
    poster = "poster"


config = Config(
    figure_size=(11, 8.5),
    figure_face_color="#CCCCCC",  # rc
    text_color="dimgrey",
    savefig_dpi=300,
    plot_face_color="#002000",  # rc
    series_colors=(  # rc.axes.prop_cycle
        "lime",
        "orange",
        "cyan",
        "magenta",
        "red",
        "blue",
        "green",
        "gray",
    ),
    legend_face_color="#001000",
    legend_edge_color="#00FF00",
    legend_text_color="#00FF00",  # Cannot be set by RC
    grid_on=True,
    grid_major_color="#00FF00",
    grid_major_alpha=0.2,
    grid_width=1.5,
    grid_minor_on=True,
    grid_minor_color="#008000",  # Cannot be set by RC
    grid_minor_alpha=0.2,  # Cannot be set by RC
)


def setup(context: Context = Context.notebook) -> None:
    """Setup matplotlib/seaborn using context and config

    :param context: Controls the size of the plot
    """

    rc = {
        "axes.axisbelow": False,
        "axes.edgecolor": "lightgrey",
        "axes.facecolor": config.plot_face_color,
        "axes.grid": config.grid_on,
        "axes.grid.axis": "both",
        "axes.grid.which": "both" if config.grid_minor_on else "major",
        "axes.labelcolor": config.text_color,
        "axes.prop_cycle": plt.cycler(color=config.series_colors),
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.facecolor": config.figure_face_color,
        "figure.figsize": config.figure_size,
        "grid.alpha": config.grid_major_alpha,
        "grid.color": config.grid_major_color,
        "grid.linestyle": "-",
        "grid.linewidth": config.grid_width,
        "legend.loc": "best",
        "legend.frameon": False,
        "legend.facecolor": config.legend_face_color,
        "legend.edgecolor": config.legend_edge_color,
        "lines.solid_capstyle": "round",
        "patch.edgecolor": "w",
        "patch.force_edgecolor": True,
        "savefig.dpi": config.savefig_dpi,
        "text.color": config.text_color,
        "xtick.bottom": False,
        "xtick.color": config.text_color,
        "xtick.direction": "out",
        "xtick.top": False,
        "ytick.color": config.text_color,
        "ytick.direction": "out",
        "ytick.left": False,
        "ytick.right": False,
        "toolbar": "None",
    }

    sns.set(
        context=context,
        palette="deep",
        font="DejaVu Sans",
        font_scale=1,
        color_codes=True,
        rc=rc,
    )


class Series:
    """All information necessary to show a set of data. """

    def __init__(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_column: str,
        color: str,
        share_x: bool = True,
        plot_type: PlotType = PlotType.line,
        # edge=None,
        # face=None,
        alpha: float = 0.75,
        marker: str = "",
        size: int = 10,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ):
        """
        :param df: xxx
        :param x_column: xxx
        :param y_column: xxx
        :param color: xxx
        :param share_x: xxx
        :param plot_type: xxx
        :param # edge=None,
        :param # face=None,
        :param alpha: xxx
        :param marker: xxx
        :param size: xxx
        :param min_size: xxx
        :param max_size: xxx
        """
        self.x = df[x_column].values
        self.y = df[y_column].values
        self.label = y_column if share_x else f"{x_column}, {y_column}"
        self.share_x = share_x

        self.plot_type = plot_type
        self.color = color
        self.alpha = alpha

        self.marker = marker
        self.size = size

        self.min_size = min_size or size
        self.max_size = max_size or size
        # self.edge = edge or color
        # self.face = face or color

        self.plotted = None

    def draw(
        self, ax: plt.Axes, show_markers: bool = False, show_values: bool = False
    ) -> None:
        """Draw the series using the appropriate plot_type,
        showing markers and values if requested

        :param ax: plot Axes
        :param show_markers: Whether to show markers
        :param show_values: Whether to show values next to point
        """

        if self.plot_type == PlotType.line:
            self.plotted = self.line(ax, show_markers)[0]
        elif self.plot_type == PlotType.scatter:
            self.plotted = self.scatter(ax)

        if show_values:
            self.display_values(ax)

    def line(self, ax: plt.Axes, show_markers: bool) -> List:
        """Draw an x/y line plot

        :param ax: Plot Axes.
        :param show_markers: Whether to show markers
        :returns: the result of ax.plot
        """

        return ax.plot(
            self.x,
            self.y,
            label=self.label,
            # color=self.color,
            alpha=self.alpha,
            marker=self.marker if show_markers else "",
            markersize=self.size,
        )

    def scatter(self, ax: plt.Axes) -> PathCollection:
        """Draw an x/y scatter plot

        :param ax: Plot Axes.
        :returns: the result of ax.scatter
        """

        return ax.scatter(
            self.x,
            self.y,
            label=self.label,
            # edgecolors=self.edge,
            # facecolor=self.face,
            alpha=self.alpha,
            marker=self.marker,
            s=np.ones_like(self.x) * (self.size * 10),  # ???
        )

    def display_values(self, ax: plt.Axes) -> None:
        """Draw series values next to points

        :param ax: Plot Axes.
        """

        xytext = (self.size, -self.size // 2)
        for xy in zip(self.x, self.y):
            ax.annotate(
                xy[1] if self.share_x else f"{xy[0]}, {xy[1]}",
                xy=xy,
                color=self.color,
                xytext=xytext,
                textcoords="offset points",
                # arrowprops={"color": "#00FF00"},
            )


class Plot:
    """All information necessary to plot the data. """

    def __init__(
        self,
        figure: plt.Figure = None,
        ax: plt.Axes = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        show_values: bool = False,
        show_markers: bool = False,
    ):
        """
        :param figure: xxx
        :param ax: xxx
        :param xlim: xxx
        :param ylim: xxx
        :param title: xxx
        :param xlabel: xxx
        :param ylabel: xxx
        :param show_values: xxx
        :param show_markers: xxx
        """
        self.figure = figure or plt.figure()
        self.ax = ax or self.figure.add_subplot(1, 1, 1)

        self.series = ()
        self.show_series = []

        self.show_values = show_values
        self.show_markers = show_markers

        self.xlim = xlim
        self.ylim = ylim
        self.xlog = False
        self.ylog = False

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.grid_on = config.grid_on

        self.ax.grid(
            which="minor",
            color=config.grid_minor_color,
            alpha=config.grid_minor_alpha,
        )

        self.mouse_handlers = MouseEventHandlers(self.ax)
        self.replace_key_handler()

    def add(self, series: List[Series]) -> None:
        """Add a series to the plot

        :param series: List of series to add to the plot
        """

        self.series = series
        self.show_series = [True for _ in self.series]

    def draw(self) -> None:
        """Draw the plot """

        ax = self.ax
        ax.clear()

        for index, s in enumerate(self.series):
            visible = self.show_series[index]
            s.draw(ax, self.show_markers, self.show_values and visible)
            s.plotted.set_visible(visible)

        if self.xlim:
            ax.set_xlim(xmin=self.xlim[0], xmax=self.xlim[1])
        if self.ylim:
            ax.set_ylim(ymin=self.ylim[0], ymax=self.ylim[1])

        if self.title:
            ax.set_title(self.title)
        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.ylabel:
            ax.set_ylabel(self.ylabel)

        if self.xlog:
            ax.set_xscale("log")
        if self.ylog:
            ax.set_yscale("log")

        if len(self.series) > 1:
            self.legend()

        self.gridlines()

    def legend(self) -> None:
        """Draw the plot's legend """

        legend = self.ax.legend()
        for text in legend.get_texts():
            text.set_color(config.legend_text_color)

    def gridlines(self) -> None:
        """Draw the plot's gridlines """

        if self.grid_on:
            self.ax.grid(True)
            self.ax.minorticks_on()
        else:
            self.ax.grid(False)
            self.ax.minorticks_off()

    def replace_key_handler(self) -> None:
        """Replace standard key and mouse handler and then show figure """

        # Remove default handlers
        canvas = self.figure.canvas
        canvas.mpl_disconnect(canvas.manager.key_press_handler_id)
        canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_key_press(self, event: KeyEvent) -> None:
        """Key handler

        :param event: KeyEvent to handle
        """

        # print(f"|{event.key}|")

        if event.key == "escape":  # Quit
            exit_cli()

        elif event.key == "enter":  # Save to image
            title = self.title or "plot"
            filename = unique_filename(f"{title}.png")
            self.figure.savefig(filename)
            print(f"Saved {filename}")

        elif event.key == "g":  # Toggle Grid
            self.grid_on = not self.grid_on
            self.gridlines()
            self.figure.canvas.draw()

        elif event.key in "123456789":  # Toggle Series Display
            n = ord(event.key) - ord("1")
            self.show_series[n] = not self.show_series[n]
            for index, s in enumerate(self.series):
                s.plotted.set_visible(self.show_series[index])
            self.figure.canvas.draw()

        elif event.key == "t":  # Cycle Plot Type
            for s in self.series:
                s.plot_type = next_item(list(PlotType), s.plot_type)

            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            self.draw()
            self.ax.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            self.figure.canvas.draw()

        elif event.key == "m":  # Toggle Series Markers
            self.show_markers = not self.show_markers
            marker = "o" if self.show_markers else ""
            for s in self.series:
                if s.plot_type in (PlotType.line,):
                    s.plotted.set_marker(marker)
            self.figure.canvas.draw()
            return

        # elif event.key == "v":  # Toggle Value Display
        #    self.show_values = not self.show_values
        #    self.draw()
        #    self.figure.canvas.draw()


def next_item(ring: List, item: Any) -> Any:
    """Find item in ring, and then
    return the next item or the first if `item` is the last in ring.
    Assumes `item` in `ring`; otherwise will throw ValueError

    :param ring: A list of items representing a ring buffer
    :param item: Item in the ring buffer
    :returns: the next item in ring buffer.
    """
    index = (ring.index(item) + 1) % len(ring)
    return ring[index]


def unique_filename(filename: Union[str, Path]) -> Path:
    """Given a filename, return a filename for a path that does not already exist.
    If the given filename does not exist it will be returned,
    otherwise a filename with
    the same stem followed by an integer and then the same suffix is returned.

    :param filename: Desired filename that the result is based on.
    :returns: a unique filename
    """
    path = Path(filename)
    if path.exists():
        stem = path.stem
        suffix = path.suffix
        for n in count(2):
            path = Path(f"{stem}{n}{suffix}")
            if not path.exists():
                break
    return path


def convert_to_label(df: pd.DataFrame, value: str) -> str:
    """Return the appropriate label for dataframe, df.
    Assumes value is either the column number (first=1)
    or is already the appropriate label.

    :param df: A pandas dataframe.
    :param value: Either a label or index for a dataframe column
    :returns: dataframe label
    """

    try:
        index = int(value) - 1
    except ValueError:
        return value

    columns = df.columns.tolist()
    if not 0 <= index < len(columns):
        exit_cli(f"Invalid column: {value}")

    return columns[index]


PlotInfo = namedtuple("PlotInfo", "xlabel ylabel series")


def load(
    df: pd.DataFrame,
    column_list: List[str],
    plot_type: PlotType = PlotType.line,
) -> PlotInfo:
    """Load dataframe into series using the columns specified by column list

    :param df: The dataframe to load from
    :param column_list: A list of labels of the columns to load
    :param plot_type:  The type to assign each series
    :returns: the xlabel, ylabel and series in a namedtuple
    """

    # assert column_list, "invalid column_list (len==0)"
    # assert len(column_list) > 1, "invalid column_list (len<=1)"
    # assert type(column_list[0]) == int, "invalid column_list (type != int)"

    share_x = True
    if not column_list:
        columns = df.columns.tolist()
        x_column = columns[0]
        y_columns = columns[1:]
        column_pairs = [(x_column, y_column) for y_column in y_columns]

    else:
        column_list_items = [len(value.split(",")) for value in column_list]
        if all((item == 1 for item in column_list_items)):
            # X Y1 Y2 ... YN
            column_list = [convert_to_label(df, value) for value in column_list]
            x_column = column_list[0]
            y_columns = column_list[1:]
            column_pairs = [(x_column, y_column) for y_column in y_columns]

        elif all((item == 2 for item in column_list_items)):
            # X1,Y1 X2,Y2 ... XN,YN
            share_x = False
            column_pairs = []
            for pair in column_list:
                x_value, y_value = pair.split(",")
                x_column = convert_to_label(df, x_value)
                y_column = convert_to_label(df, y_value)
                column_pairs.append((x_column, y_column))

        else:
            exit_cli("ERROR")

    # Verify all column indices are valid
    for x_column, y_column in column_pairs:
        if x_column not in df:
            exit_cli(f"Invalid column: {x_column}")
        if y_column not in df:
            exit_cli(f"Invalid column: {y_column}")

    cycle = plt.rcParams["axes.prop_cycle"]()

    series = [
        Series(
            df,
            x_column,
            y_column,
            plot_type=plot_type,
            marker="o",
            share_x=share_x,
            color=next(cycle).get("color"),
        )
        for x_column, y_column in column_pairs
    ]

    return PlotInfo(
        xlabel=column_pairs[0][0] if share_x or len(column_pairs) == 1 else "",
        ylabel=column_pairs[0][1] if len(column_pairs) == 1 else "",
        series=series,
    )


def exit_cli(comment: Optional[str] = None) -> None:
    """Exit using typer, echoing comment if provided

    :param comment: String to print before exiting
    """
    if comment:
        typer.echo(comment)
    sys.exit(0)


def version_option() -> bool:
    """
    :returns: the typer Option that handles --version
    """

    def version_callback(_ctxt: typer.Context, value: bool):
        if value:
            exit_cli(f"plot version: {__version__}")

    return typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the version and exit.",
    )


def run(
    data_file_path: Path = typer.Argument(default=None, exists=True, dir_okay=False),
    columns: List[str] = typer.Argument(default=None),
    plot_type: PlotType = typer.Option(PlotType.line, "--type"),
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    context: Context = typer.Option(Context.notebook),
    head: bool = typer.Option(None, help="Display head of data file."),
    demo: bool = typer.Option(None, help="Generate demo.png and use it."),
    version: bool = version_option(),
) -> None:
    """
        Plot will read the DATA_FILE_PATH and plot the data specified by COLUMNS.

        COLUMNS must be of the form
          X Y1 Y2 ... Yn
        or
          X1,Y1 X2,Y2 ... Xn,Yn

        The column value must either be the index of the column (1..N),
        or the name of the column.


        User Interface

    \b
        Key    | Result
        -------|------------------------
        g      | Toggle Grid
        t      | Cycle Plot Type
        m      | Toggle Series Markers
        1-9    | Toggle Series 1-9 Display
        enter  | Save Plot to png Image
        escape | Exit

    \b
        Holding the left mouse button down and moving the mouse will pan the plot.
        Rolling the mouse wheel up and down will zoom out and in where the mouse is.
    """

    if data_file_path:
        delimiter = None
        header = 0  # int = typer.Option(0, help="Number of rows in data_file header"),
        df = pd.read_csv(
            data_file_path,
            header=header,
            engine="python",
            sep=delimiter,
        )
        title = title or data_file_path.stem.replace("_", " ")
    elif demo:
        df = demo_df("demo.dat")
        title = title or "Demo"
    else:
        exit_cli("Must specify data file path or use --demo")

    if head:
        exit_cli(df.head(10))

    setup(context)

    plot_info = load(df, columns, plot_type=plot_type)
    plot1 = Plot(
        title=title or title,
        xlabel=xlabel or plot_info.xlabel,
        ylabel=ylabel or plot_info.ylabel,
    )
    plot1.add(plot_info.series)
    plot1.draw()
    plt.show()


def main() -> None:
    """Call the app command run """

    app = typer.Typer(add_completion=False)
    app.command()(run)
    app()


if __name__ == "__main__":
    main()
