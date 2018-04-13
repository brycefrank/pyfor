from plotly import offline
from plotly.graph_objs import graph_objs as go
import numpy as np


def iplot3d(las, max_points, point_size):
    """
    Plots the 3d point cloud in a compatible version for Jupyter notebooks.
    :return:
    # TODO refactor to a name that isn't silly
    """
    # Check if in iPython notebook
    try:
        cfg = get_ipython().config
        if 'jupyter' in cfg['IPKernelApp']['connection_file']:
            if las.header.count > max_points:
                print("Point cloud too large, down sampling for plot performance.")
                rand = np.random.randint(0, las.header.count, 30000)
                x = las.x[rand]
                y = las.y[rand]
                z = las.z[rand]

                trace1 = go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=z,
                        colorscale='Viridis',
                        opacity=1
                    )
                )

                data = [trace1]
                layout = go.Layout(
                    margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0
                    ),
                    scene=dict(
                        aspectmode="data"
                    )
                )
                offline.init_notebook_mode(connected=True)
                fig = go.Figure(data=data, layout=layout)
                offline.iplot(fig)
        else:
            print("This function can only be used within a Jupyter notebook.")
            return(False)
    except NameError:
        return(False)
