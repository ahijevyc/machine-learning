import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pdb
from sklearn.preprocessing import LabelEncoder
cmap_data = plt.cm.Set2
def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    le = LabelEncoder().fit(y) # convert str labels to integers
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        sc = ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=plt.cm.Set1, norm=matplotlib.colors.BoundaryNorm([0,1-1/n_splits,1.0001],2)
        )
        # TODO: make sure class label percentages are in correct order
        ax.text(len(indices),ii+0.5,f" {len(tr)/len(X)*100:2.0f}% train / {len(tt)/len(X)*100:2.0f}% test\n{(y.iloc[tr].value_counts()[le.classes_]/len(tr)*100).round().astype(int).values} {(y.iloc[tt].value_counts()[le.classes_]/len(tt)*100).round().astype(int).values}", va="center")
    cbar = plt.colorbar(sc, location="top", shrink=0.15, anchor=(0.94,0), ticks=[(1-1/n_splits)/2,1-1/2/n_splits], spacing='proportional')
    cbar.ax.set_xticklabels(['training','test'])
    cbar.outline.set_visible(False)

    # Plot the data classes and groups at the end
    nclass = len(le.classes_)
    boundaries = np.arange(nclass+1)-0.5
    norm = matplotlib.colors.BoundaryNorm(boundaries, nclass)
    sc = ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=le.transform(y), marker="_", lw=lw, cmap=cmap_data, norm=norm
    )
    # get height of linewidth (lw) points in axes units
    height = (ax.transAxes.inverted().transform([0,lw]) - ax.transAxes.inverted().transform([0,0]))[1]
    cax = ax.inset_axes([1.002, ax.transLimits.transform([0,1.5])[1]-height, 0.1, height])
    cbar = plt.colorbar(sc, ax=ax, cax=cax, orientation="horizontal", ticks=np.arange(nclass))
    cbar.ax.set_xticklabels(le.inverse_transform(range(nclass)))
    cbar.outline.set_visible(False)
    if np.unique(group).size > len(cmap_data.colors):
        print("more groups than colors")
        group = group % len(cmap_data.colors)

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=plt.cm.Paired
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["label", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, len(y)],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax

