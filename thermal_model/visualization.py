"""Visualization utilities for the thermal model."""
from typing import Optional

import numpy as np
import plotly.graph_objects as go


def plot_temperature_history(history: np.ndarray, save_path: Optional[str] = None):
    """Plot temperature profile over time.

    Parameters
    ----------
    history : np.ndarray
        2-D array of shape (time_steps, num_zones).
    save_path : str | None
        If provided, save the figure to an HTML file; otherwise display it.
    """
    if history.ndim != 2:
        raise ValueError("history must be 2-D array (time, zones)")

    time = np.arange(history.shape[0])
    fig = go.Figure()
    for zone in range(history.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=history[:, zone],
                mode="lines",
                name=f"Zone {zone}",
            )
        )

    fig.update_layout(
        title="Battery Temperature History",
        xaxis_title="Time (s)",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved temperature plot to {save_path}")
    else:
        fig.show() 