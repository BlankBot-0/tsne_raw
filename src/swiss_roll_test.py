import numpy as np
import pandas as pd
from tsne import *

# make a swiss roll

Es = np.array([[7.5, 7.5], [7.5, 12.5], [12.5, 7.5], [12.5, 12.5]])
dfs = []
swiss_roll = np.zeros((1, 3))
for i, E in enumerate(Es):
    XY = np.random.normal(E, 1, size=(400, 2))
    XYZ = np.zeros((400, 3))
    XYZ[:, 0] = XY[:, 0] * np.cos(XY[:, 0])
    XYZ[:, 1] = XY[:, 1]
    XYZ[:, 2] = XY[:, 0] * np.sin(XY[:, 0])
    swiss_roll = np.concatenate([swiss_roll, XYZ])
    df = pd.DataFrame(XYZ, columns=['x', 'y', 'z'])
    df['label'] = [i]*400
    dfs.append(df)
swiss_roll = swiss_roll[1:, :]

swiss_roll_df = pd.concat(dfs)
# fig = px.scatter_3d(swiss_roll_df, x="x", y="y", z='z', color="label")
# fig.show()

# Create initialization function
def rnd_gaussian_init(X, spec_idxs, spec_coords, rng, dim, E, D):
    Y = rng.normal(E, D, [X.shape[0], dim])
    for i in spec_idxs:
        Y[i] = spec_coords[i]
    return Y

# Set global parameters
PERPLEXITY = 250
SEED = 265                 # Random seed
TSNE = True                # If False, Symmetric SNE


# numpy RandomState for reproducibility
rng = np.random.RandomState(SEED)

# labels
y = [0]*400 + [1]*400 + [2]*400 + [3]*400

# fix group 0
spec_idxs = np.arange(400)
spec_coords = rng.normal([7.5, 7.5], 1, size=(400, 2))

# Obtain matrix of joint probabilities p_ij
P = p_joint(swiss_roll, PERPLEXITY)

# Fit SNE or t-SNE
Y, df = estimate_sne(swiss_roll, y, P,
                 rng,
                 num_iters=4000,
                 q_fn=q_tsne,
                 grad_fn=tsne_grad,
                 init_fn=rnd_gaussian_init,
                 dim=2,
                 learning_rate=20,
                 momentum=0.9,
                 plot=40,
                 spec_idxs=spec_idxs,
                 spec_coords=spec_coords,
                 E=[10, 10],
                 D=1)