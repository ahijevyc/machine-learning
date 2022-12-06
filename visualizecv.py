import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pdb
from sklearn.preprocessing import LabelEncoder
cmap_data = plt.cm.Set2



def plot_cv_indices(cv, X, y, group, ax):
    """Create a sample plot for indices of a cross-validation object."""

    n_splits = cv.n_splits
    le = LabelEncoder().fit(y)  # convert str labels to integers
    xmin = 0
    dx = len(X)
    dy = 0.96

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        logging.info(f"{ii} - {len(tr)} train {len(tt)} test")
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        ymin = ii
        rect = [xmin, ymin, dx, dy]
        iax = ax.inset_axes(rect, transform=ax.transData)
        iax.axis('off')

        # Visualize the results
        img = iax.imshow(indices[np.newaxis], aspect="auto",
                         cmap=plt.cm.Set1, norm=BoundaryNorm(
                             [0, 1-1/n_splits, 1.0001], 2)
                         )
        # TODO: make sure class label percentages are in correct order
        txt = f"{len(tr)/len(X):2.0%} train / {len(tt)/len(X):2.0%} test\n"
        txt += " ".join([f"{x:.0%}" for x in y.iloc[tr].value_counts()/len(tr)])
        txt += " / "
        txt += " ".join([f"{x:.0%}" for x in y.iloc[tt].value_counts()/len(tt)])
        
        ax.text(len(indices)*1.01, ii+0.5, txt, va="center", fontsize="xx-small")
    cbar = plt.colorbar(img, location="top", shrink=0.15, anchor=(0.94, 0), ticks=[
                        (1-1/n_splits)/2, 1-1/2/n_splits], spacing='proportional', ax=ax)
    cbar.ax.set_xticklabels(['training', 'test'])
    cbar.ax.tick_params(labelsize='xx-small')
    cbar.outline.set_visible(False)


    # Plot the data classes and groups at the end
    nclass = len(le.classes_)
    boundaries = np.arange(nclass+1)-0.5
    norm = BoundaryNorm(boundaries, nclass)
    rect = [0, ii+1, dx, dy]
    iax = ax.inset_axes(rect, transform=ax.transData)
    iax.axis('off')
    img = iax.imshow(le.transform(y)[np.newaxis], aspect="auto",
                     cmap=cmap_data, norm=norm, interpolation='none'
                     )

    rect = [len(X)*1.01, ii+1, len(X)*0.2, dy]
    cax = ax.inset_axes(rect, transform=ax.transData)
    cbar = plt.colorbar(img, ax=ax, cax=cax,
                        orientation="horizontal", ticks=np.arange(nclass))
    cbar.ax.set_xticklabels(le.inverse_transform(range(nclass)))
    cbar.outline.set_visible(False)
    
    if group is not None:
        if np.unique(group).size > len(cmap_data.colors):
            logging.warning(
                f"more groups ({np.unique(group).size}) than colors ({len(cmap_data.colors)})")
            group = group % len(cmap_data.colors)

        rect = [0, ii+2, dx, dy]
        iax = ax.inset_axes(rect, transform=ax.transData)
        iax.axis('off')
        img = iax.imshow(group[np.newaxis], aspect="auto",
                         cmap=plt.cm.Paired)
        
    # Formatting
    yticklabels = list(range(n_splits)) + ["label", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[xmin, xmin+dx],
    )
    ax.set_title(type(cv).__name__, fontsize=15)
    
    return ax
