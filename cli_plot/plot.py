#!/usr/bin/env python

"""
TODO
"""

from typing import List, Union, Any
from pathlib import Path
from enum import Enum
from collections import namedtuple
from itertools import count

import typer
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PathCollection

from .pyplotui import PyPlotUI
from .config import Config
from .functions import sine_wave, cosine_wave, normal, damped_sine_wave, dataframe

__version__ = "0.0.1"


class PlotType(str, Enum):
    """ TODO """

    line = "line"
    scatter = "scatter"


class Context(str, Enum):
    """ Seaborn context set the size of the plot  """

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
    legend_text_color="#00FF00",  # TODO: Cannot be set by RC - do by plot
    grid_on=True,
    grid_major_color="#00FF00",
    grid_major_alpha=0.2,
    grid_width=1.5,
    grid_minor_on=True,
    grid_minor_color="#008000",  # Cannot be set by RC
    grid_minor_alpha=0.2,  # Cannot be set by RC
)


def setup(context: Context = Context.notebook) -> None:
    """ TODO """

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
    """ TODO """

    def __init__(
        self,
        df,
        x_column,
        y_column,
        color,
        share_x=True,
        plot_type=PlotType.line,
        # edge=None,
        # face=None,
        alpha=0.75,
        marker="",
        size=10,
        min_size=None,
        max_size=None,
    ):
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

    def plot(self, ax, show_markers=False, show_values=False) -> None:
        """ TODO """

        if self.plot_type == PlotType.line:
            self.plotted = self.line(ax, show_markers)[0]
        elif self.plot_type == PlotType.scatter:
            self.plotted = self.scatter(ax)

        if show_values:
            self.display_values(ax)

    def line(self, ax, show_markers) -> List:
        """ TODO """

        return ax.plot(
            self.x,
            self.y,
            label=self.label,
            # color=self.color,
            alpha=self.alpha,
            marker=self.marker if show_markers else "",
            markersize=self.size,
        )

    def scatter(self, ax) -> PathCollection:
        """ TODO """

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

    def display_values(self, ax) -> None:
        """ TODO """

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
    """ TODO """

    def __init__(
        self,
        figure=None,
        ax=None,
        xlim=None,
        ylim=None,
        title=None,
        xlabel=None,
        ylabel=None,
        show_values=False,
        show_markers=False,
    ):
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

    def add(self, series) -> None:
        """ TODO """

        self.series = series
        self.show_series = [True for _ in self.series]

    def draw(self) -> None:
        """ TODO """

        ax = self.ax
        ax.clear()

        for index, s in enumerate(self.series):
            s.plot(ax, self.show_markers, self.show_values and self.show_series[index])
            s.plotted.set_visible(self.show_series[index])

        if self.xlim:
            ax.set_xlim(self.xlim)
        if self.ylim:
            ax.set_ylim(self.ylim)

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
        """ TODO """

        legend = self.ax.legend()
        for text in legend.get_texts():
            text.set_color(config.legend_text_color)

    def gridlines(self) -> None:
        """ TODO """

        if self.grid_on:
            self.ax.minorticks_on()
            self.ax.grid(
                which="minor",
                color=config.grid_minor_color,
                alpha=config.grid_minor_alpha,
            )
        else:
            self.ax.grid(False)
            self.ax.minorticks_off()

    def post(self) -> None:
        """ TODO """

        figure = self.figure
        canvas = figure.canvas

        # Remove default handlers
        canvas.mpl_disconnect(canvas.manager.key_press_handler_id)
        canvas.mpl_connect("key_press_event", self._on_key_press)
        ui = PyPlotUI(figure, self.ax)
        plt.show()

    def _on_key_press(self, event) -> None:
        """ TODO """

        # print(f"|{event.key}|")

        if event.key == "g":  # Toggle Grid
            self.grid_on = not self.grid_on
            self.gridlines()
            self.figure.canvas.draw()
            return

        if event.key == "t":  # Cycle Plot Type
            for s in self.series:
                s.plot_type = next_item(list(PlotType), s.plot_type)

            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            self.draw()
            self.ax.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

            self.figure.canvas.draw()
            return

        if event.key == "m":  # Toggle Series Markers
            self.show_markers = not self.show_markers
            marker = "o" if self.show_markers else ""
            for s in self.series:
                if s.plot_type in (PlotType.line,):
                    s.plotted.set_marker(marker)
            self.figure.canvas.draw()
            return

        """
        if event.key == "v":  # Toggle Value Display
            self.show_values = not self.show_values
            self.draw()
            self.figure.canvas.draw()
            return
        """

        if event.key in "123456789":  # Toggle Series Display
            n = ord(event.key) - ord("1")
            self.show_series[n] = not self.show_series[n]
            for index, s in enumerate(self.series):
                s.plotted.set_visible(self.show_series[index])
            self.figure.canvas.draw()
            return

        if event.key == "enter":  # Save to image
            title = self.title or "plot"
            filename = unique_filename(f"{title}.png")
            self.figure.savefig(filename)
            print(f"Saved {filename}")
            return

        if event.key == "escape":  # Quit
            exit(0)

        # plt.close()
        return


def next_item(ring: List, item: Any) -> Any:
    """
    Returns the next item in ring buffer,
    returning the first item if `item` is the last
    Assumes `item` in `ring`; otherwise will throw ValueError
    """
    index = (ring.index(item) + 1) % len(ring)
    return ring[index]


def unique_filename(filename: Union[str, Path]) -> Path:
    path = Path(filename)
    if path.exists():
        stem = path.stem
        suffix = path.suffix
        for n in count(2):
            path = Path(f"{stem}{n}{suffix}")
            if not path.exists():
                break
    return path


def convert_to_label(columns: List[str], value: str) -> str:
    """ TODO """

    try:
        index = int(value) - 1
    except ValueError:
        return value

    if not 0 <= index < len(columns):
        print(f"Invalid column: {value}")
        raise typer.Exit()

    return columns[index]


PlotInfo = namedtuple("PlotInfo", "xlabel ylabel series")


def demo_df() -> pd.DataFrame:
    """ TODO """

    t = np.linspace(start=-3, stop=3, num=200)
    df = dataframe(
        time=t,
        sin=sine_wave(t, 1.0),
        cos=cosine_wave(t, 1.0),
        normal=normal(t),
        damp=damped_sine_wave(t, amp=0.5, damp=0.5),
    )

    df.to_csv("demo.dat", index=False)
    return df


def load(df, column_list: List[str], plot_type: PlotType = PlotType.line,) -> PlotInfo:
    """ TODO """

    # assert column_list, "invalid column_list (len==0)"
    # assert len(column_list) > 1, "invalid column_list (len<=1)"
    # assert type(column_list[0]) == int, "invalid column_list (type != int)"

    columns = df.columns.tolist()

    share_x = True
    if not column_list:
        x_column = columns[0]
        y_columns = columns[1:]
        column_pairs = [(x_column, y_column) for y_column in y_columns]

    else:
        column_list_items = [len(value.split(",")) for value in column_list]
        if all((item == 1 for item in column_list_items)):
            # X Y1 Y2 ... YN
            column_list = [convert_to_label(columns, value) for value in column_list]
            x_column = column_list[0]
            y_columns = column_list[1:]
            column_pairs = [(x_column, y_column) for y_column in y_columns]

        elif all((item == 2 for item in column_list_items)):
            # X1,Y1 X2,Y2 ... XN,YN
            share_x = False
            column_pairs = []
            for pair in column_list:
                x_column, y_column = pair.split(",")
                x_column = convert_to_label(columns, x_column)
                y_column = convert_to_label(columns, y_column)
                column_pairs.append((x_column, y_column))

        else:
            print("ERROR")
            raise typer.Exit()

    # TODO: Verify all column indices are valid

    cycle = plt.rcParams["axes.prop_cycle"]()

    for x_column, y_column in column_pairs:
        if x_column not in df:
            print(f"Invalid column: {x_column}")
            raise typer.Exit()
        if y_column not in df:
            print(f"Invalid column: {y_column}")
            raise typer.Exit()

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


def version_option() -> bool:
    """ TODO """

    def version_callback(_ctxt, value: bool):
        if value:
            typer.echo(f"plot version: {__version__}")
            raise typer.Exit()

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

    The column value must be either the index of the column 1..N, or the name of the column.


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
    Rolling the mouse wheel up and down will zoom the plot where the mouse is located.
    """

    if data_file_path:
        delimiter = None
        header = 0  # int = typer.Option(0, help="Number of rows in data_file header"),
        df = pd.read_csv(data_file_path, header=header, engine="python", sep=delimiter,)
        title = title or data_file_path.stem.replace("_", " ")
    elif demo:
        df = demo_df()
        title = title or "Demo"
    else:
        print("Must specify data file path or use --demo")
        raise typer.Exit()

    if head:
        print(df.head(10))
        exit(0)

    setup(context)

    plot_info = load(df, columns, plot_type=plot_type)
    plot1 = Plot(
        title=title or title,
        xlabel=xlabel or plot_info.xlabel,
        ylabel=ylabel or plot_info.ylabel,
    )
    plot1.add(plot_info.series)
    plot1.draw()
    plot1.post()


def main() -> None:
    """ TODO """

    app = typer.Typer(add_completion=False)
    app.command()(run)
    app()


if __name__ == "__main__":
    main()
