"""
Functions to animate gradient descent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def add_step_to_scatter(X, class_idxs, i, df):
    axes = ['x', 'y', 'z']
    newdf = pd.DataFrame(X, columns=[axes[i] for i in range(X.shape[1])])
    newdf['iter'] = [i]*len(X)
    newdf['cat'] = class_idxs
    return pd.concat([df, newdf], ignore_index=True)


def show_2d_plot(df):
    fig = px.scatter(df, x="x", y="y", animation_frame="iter", color="cat",
                 width=800, height=800)
    fig.show()


def show_3d_plot(df):
    fig = px.scatter_3d(df, x="x", y="y", z='z', animation_frame="iter", color="cat",
                 width=800, height=800)
    fig.show()
