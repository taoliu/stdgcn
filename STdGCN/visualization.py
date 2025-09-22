import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from tqdm.notebook import tqdm


def _marker_radius_in_data_units(ax, pt_size):
    """Convert scatter-style marker size to an approximate radius in data units."""

    area_points = pt_size ** 2
    radius_points = np.sqrt(area_points / np.pi)
    fig = ax.figure
    radius_pixels = radius_points * fig.dpi / 72.0

    transform = ax.transData
    inv_transform = transform.inverted()

    origin_display = transform.transform((0.0, 0.0))
    x_display = origin_display + np.array([radius_pixels, 0.0])
    y_display = origin_display + np.array([0.0, radius_pixels])

    dx = inv_transform.transform(x_display)[0] - inv_transform.transform(origin_display)[0]
    dy = inv_transform.transform(y_display)[1] - inv_transform.transform(origin_display)[1]

    return float(min(abs(dx), abs(dy)))


def _build_pie_polygons(dist, xpos, ypos, radius, colors):
    """Return polygon vertices and facecolors for a single pie chart."""

    dist = np.asarray(dist, dtype=float)
    total = dist.sum()
    if total <= 0:
        return []

    normalized = np.cumsum(dist) / total
    starts = np.concatenate(([0.0], normalized[:-1]))

    polygons = []
    for idx, (start, end, value) in enumerate(zip(starts, normalized, dist)):
        if value <= 0:
            continue

        theta_start = start * 2 * np.pi
        theta_end = end * 2 * np.pi

        # ensure at least a triangle, add resolution for larger wedges
        resolution = max(4, int(np.ceil((theta_end - theta_start) / (np.pi / 24))))
        angles = np.linspace(theta_start, theta_end, resolution)
        xs = xpos + radius * np.cos(angles)
        ys = ypos + radius * np.sin(angles)

        verts = np.column_stack([
            np.concatenate(([xpos], xs, [xpos])),
            np.concatenate(([ypos], ys, [ypos]))
        ])
        polygons.append((verts, colors[idx]))

    return polygons


def plot_frac_results(predict, cell_type_list, coordinates,
                      file_name=None,
                      point_size=20,
                      if_show=True,
                      color_dict=None,
                      fig_height=32,
                      fig_width=18
                      ):

    coordinates.columns = ['coor_X', 'coor_Y']
    labels = cell_type_list
    if color_dict is not None:
        colors = []
        for i in cell_type_list:
            colors.append(color_dict[i])
    else:
        if len(labels) <= 10:
            colors = plt.rcParams["axes.prop_cycle"].by_key()['color'][:len(labels)]
        else:
            import matplotlib
            color = plt.get_cmap('rainbow', len(labels))
            colors = []
            for x in color([range(len(labels))][0]):
                colors.append(matplotlib.colors.to_hex(x, keep_alpha=False))

    # str_len = 0
    # for item in cell_type_list:
    #     str_len = max(str_len, len(item))
    # extend_region = str_len/15 + 3

    fig, ax = plt.subplots(figsize=(fig_height, fig_width), dpi=72)

    coords_array = coordinates[['coor_X', 'coor_Y']].values
    xmin, xmax = coords_array[:, 0].min(), coords_array[:, 0].max()
    ymin, ymax = coords_array[:, 1].min(), coords_array[:, 1].max()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    radius = _marker_radius_in_data_units(ax, point_size)
    if radius <= 0:
        span = max(xmax - xmin, ymax - ymin)
        radius = span * 0.01 if span > 0 else 1.0
    padding = radius if radius > 0 else 0.0
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_ylim(ymin - padding, ymax + padding)
    polygons = []
    facecolors = []
    colors_rgba = [mcolors.to_rgba(c) for c in colors]

    for idx in tqdm(range(predict.shape[0]), desc="Plotting pie plots:"):
        slices = _build_pie_polygons(predict[idx],
                                     coords_array[idx, 0],
                                     coords_array[idx, 1],
                                     radius,
                                     colors_rgba)
        if not slices:
            continue
        for verts, color in slices:
            polygons.append(verts)
            facecolors.append(color)

    if polygons:
        collection = PolyCollection(polygons,
                                    facecolors=facecolors,
                                    edgecolors='none',
                                    linewidths=0)
        ax.add_collection(collection)

    patches = [mpatches.Patch(color=colors[i],
                              label="{:s}".format(labels[i])
                              ) for i in range(len(colors))]
    fontsize = max(predict.shape[0]/100, 10)
    fontsize = min(fontsize, 30)
    ax.legend(handles=patches, fontsize=fontsize, bbox_to_anchor=(1, 1), loc="upper left")
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name,
                    dpi=72,
                    # bbox_inches='tight'
                    )
    if if_show:
        plt.show()
    plt.close('all')


def plot_scatter_by_type(predict,
                         cell_type_list,
                         coordinates,
                         point_size=20,
                         file_path=None,
                         if_show=True,
                         fig_height=32,
                         fig_width=18):

    coordinates.columns = ['coor_X', 'coor_Y']

    for i in tqdm(range(len(cell_type_list)), desc="Plotting cell type scatter plot:"):

        fig, ax = plt.subplots(figsize=(fig_height, fig_width), dpi=72)#figsize=(len(coordinates['coor_X'].unique())*point_size*size_coefficient+1, len(coordinates['coor_Y'].unique())*point_size*size_coefficient))
        cm = plt.cm.get_cmap('YlOrRd')
        ax = plt.scatter(coordinates['coor_X'], coordinates['coor_Y'],
                         s=point_size**2,
                         vmin=0, vmax=0.8,
                         c=predict[:, i], cmap=cm)

        cbar = plt.colorbar(ax, fraction=0.05)
        labelsize = max(predict.shape[0]/100, 10)
        labelsize = min(labelsize, 30)
        cbar.ax.tick_params(labelsize=labelsize)
        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        # plt.xlim(coordinates['coor_X'].min()-0.5, coordinates['coor_X'].max()+0.5)
        # plt.ylim(coordinates['coor_Y'].min()-0.5, coordinates['coor_Y'].max()+0.5)
        plt.tight_layout()
        if file_path is not None:
            name = cell_type_list[i].replace('/', '_')
            plt.savefig(file_path+'/{}.jpg'.format(name), dpi=72, bbox_inches='tight')
        if if_show:
            plt.show()
        plt.close('all')
