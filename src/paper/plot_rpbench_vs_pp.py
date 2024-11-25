import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import numpy as np
import fire  # type: ignore
from scipy.stats import spearmanr  # type: ignore


def main(input_path: str, output_path: str) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = ":"

    # Read the CSV data
    df = pd.read_csv(input_path)

    print(spearmanr(df["pp_score"], df["rpbench_score"]))

    # Calculate ranks for pp_score and cw_score (lower is better)
    df["pp_rank"] = df["pp_score"].rank(ascending=False)
    df["rpbench_rank"] = df["rpbench_score"].rank(ascending=False)

    # Sort the dataframe by pp_rank
    df_sorted = df.sort_values("pp_rank")

    # Create the plot with adjusted dimensions
    fig, ax = plt.subplots(figsize=(12, 7))  # Increased size for better visibility

    # Create a custom color palette
    colors = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#a65628",
        "#f781bf",
        "#999999",
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
    ]

    # Define line styles and markers
    line_styles = ["-", "--", "-.", ":"]
    markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "<",
        ">",
        "p",
        "*",
        "h",
        "H",
        "+",
        "x",
        "d",
        "|",
        "_",
    ]

    # Ensure we have enough colors, line styles, and markers
    num_models = len(df_sorted)
    colors = colors * (num_models // len(colors) + 1)
    line_styles = ["-" for c in colors]
    markers = markers * (num_models // len(markers) + 1)

    # Plot lines for each model
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        color = colors[i]
        line_style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]
        plt.plot(
            [0, 1],
            [row["rpbench_rank"], row["pp_rank"]],
            color=color,
            linestyle=line_style,
            linewidth=2,
            marker=marker,
            markersize=6,
            markeredgewidth=2,
            markeredgecolor="black",
            markerfacecolor=color,
            label=f"{row['model']}",
        )

    # Customize the plot
    plt.title("Model Rankings: RPBench vs PingPong", fontsize=18)
    plt.xlabel("Benchmark", fontsize=16)
    plt.ylabel("Rank (higher position = better performance)", fontsize=16)

    # Set x-axis to show only PingPong and Creative Writing
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["RPBench", "PingPong"], fontsize=14)

    # Set y-axis to show only integer values and flip the axis
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim(ax.get_ylim()[::-1])  # Flip y-axis

    # Add vertical lines to separate PingPong and Creative Writing
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.5)

    # Add legend with larger font size and adjust its position
    plt.legend(
        bbox_to_anchor=(1.05, 0.973),
        loc="upper left",
        fontsize=13,
        borderaxespad=0.0,
        labelspacing=1.2425,
    )

    # Adjust layout
    plt.tight_layout()

    # Save the figure as a high-resolution PNG and PDF
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    fire.Fire(main)
