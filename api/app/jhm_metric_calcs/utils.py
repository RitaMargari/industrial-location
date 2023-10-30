import networkit as nk
import pandas as pd
import geopandas as gpd
import networkx as nx
import math
import shapely
import numpy as np
import requests


def get_nx2_nk_idmap(G_nx):
    """
    Взято из библиотеки
    """
    idmap = dict(
        (id, u) for (id, u) in zip(G_nx.nodes(), range(G_nx.number_of_nodes()))
    )
    return idmap


def get_nk_attrs(G_nx):
    """
    Взято из библиотеки
    """

    attrs = dict(
        (u, {"x": d[-1]["x"], "y": d[-1]["y"]})
        for (d, u) in zip(G_nx.nodes(data=True), range(G_nx.number_of_nodes()))
    )

    return attrs


def convert_nx2nk(G_nx, idmap=None, weight=None):
    """
    Взято из библиотеки
    """

    if not idmap:
        idmap = get_nx2_nk_idmap(G_nx)
    n = max(idmap.values()) + 1
    edges = list(G_nx.edges())

    if weight:
        G_nk = nk.Graph(n, directed=G_nx.is_directed(), weighted=True)
        for u_, v_ in edges:
            u, v = idmap[u_], idmap[v_]
            d = dict(G_nx[u_][v_])
            if len(d) > 1:
                for d_ in d.values():
                    v__ = G_nk.addNodes(2)
                    u__ = v__ - 1
                    w = round(d_[weight], 1) if weight in d_ else 1
                    G_nk.addEdge(u, v, w)
                    G_nk.addEdge(u_, u__, 0)
                    G_nk.addEdge(v_, v__, 0)
            else:
                d_ = list(d.values())[0]
                w = round(d_[weight], 1) if weight in d_ else 1
                G_nk.addEdge(u, v, w)
    else:
        G_nk = nk.Graph(n, directed=G_nx.is_directed())
        for u_, v_ in edges:
            u, v = idmap[u_], idmap[v_]
            G_nk.addEdge(u, v)

    return G_nk


def get_nk_distances(nk_dists, source_nodes, target_node):
    distances = [nk_dists.getDistance(target_node, node) for node in source_nodes]
    return distances


def fix_crs(gdf):
    gdf_crs = gdf.crs.to_epsg()

    if gdf_crs == 4326:
        gdf = gdf.to_crs(gdf.estimate_utm_crs().to_epsg())
    else:
        pass

    return gdf


def G_to_gdf(G: nx.DiGraph):
    graph_crs = G.graph["crs"]

    graph_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
    graph_gdf = gpd.GeoDataFrame(
        graph_df,
        geometry=gpd.points_from_xy(graph_df["x"], graph_df["y"]),
        crs=graph_crs,
    )

    graph_gdf = fix_crs(graph_gdf)
    G_nx = G.subgraph(graph_gdf.index)
    G_nx2 = nx.convert_node_labels_to_integers(G_nx)

    graph_df2 = pd.DataFrame.from_dict(dict(G_nx2.nodes(data=True)), orient="index")
    graph_gdf2 = gpd.GeoDataFrame(
        graph_df2,
        geometry=gpd.points_from_xy(graph_df2["x"], graph_df2["y"]),
        crs=graph_crs,
    )

    graph_gdf2 = fix_crs(graph_gdf2)

    return graph_gdf2, G_nx2


def convert_wgs_to_utm(lon: float, lat: float):
    """Based on lat and lng, return best utm epsg-code"""
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = "0" + utm_band
    if lat >= 0:
        epsg_code = "326" + utm_band
        return epsg_code
    epsg_code = "327" + utm_band
    return epsg_code


def get_nearest_nodes(graph_gdf: gpd.GeoDataFrame, locations: gpd.GeoSeries):
    return graph_gdf["geometry"].sindex.nearest(
        locations, return_distance=False, return_all=False
    )


def _dissolve_by_grid(
    grid: gpd.GeoDataFrame,
    gdf: gpd.GeoDataFrame,
    cols=["Iq"],
    aggfunc: str = "mean",
    dropna: bool = True,
):
    merged = gpd.sjoin(gdf, grid, how="left", predicate="within")
    dissolve = merged.dissolve(by="index_right", aggfunc=aggfunc)
    for col in cols:
        grid.loc[dissolve.index, f"{col}_{aggfunc}"] = dissolve[col].values
    if dropna:
        grid.dropna(inplace=True)
    grid = grid.round(3)
    return grid


from shapely.geometry import box

def create_grid(gdf, cols, cell_size_meters):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    # determine cell size in meters
    xdist = xmax - xmin
    n_cells = int(xdist / cell_size_meters)
    cell_size = xdist / n_cells
    # projection of the grid
    crs = gdf.crs
    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax + cell_size, cell_size):
        for y0 in np.arange(ymin, ymax + cell_size, cell_size):
            # bounds
            x1 = x0 + cell_size
            y1 = y0 - cell_size
            grid_cells.append(box(x0, y0, x1, y1))
    grid = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs=crs)

    grid = _dissolve_by_grid(grid, gdf, cols=cols, aggfunc="mean", dropna=True)
    return grid


def read_intermodal_G_from_gdrive() -> nx.DiGraph:
    # link to intermodal graph (without car transport type) to Gosha's private gdrive folder
    url = 'https://drive.google.com/file/d/1vGGh1s7EIjxgGEF_0Dylb6XNBnCpApEQ/view?usp=sharing'
    response = requests.get(url, timeout=300)
    output = response.content

    # Create nx graph object
    graph = nx.read_adjlist(output.splitlines())
    return graph
