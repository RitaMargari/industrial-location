import networkit as nk
import pandas as pd
import geopandas as gpd
import networkx as nx


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

def G_to_gdf(G: nx.DiGraph, local_crs: int):
    graph_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
    graph_gdf = gpd.GeoDataFrame(
        graph_df,
        geometry=gpd.points_from_xy(graph_df["x"], graph_df["y"]),
        crs=local_crs,
    )

    G_nx = G.subgraph(graph_gdf.index)
    G_nx2 = nx.convert_node_labels_to_integers(G_nx)

    graph_df2 = pd.DataFrame.from_dict(dict(G_nx2.nodes(data=True)), orient="index")
    graph_gdf2 = gpd.GeoDataFrame(
        graph_df2,
        geometry=gpd.points_from_xy(graph_df2["x"], graph_df2["y"]),
        crs=local_crs,
    )

    return graph_gdf2, G_nx2


def get_nearest_nodes(graph_gdf: gpd.GeoDataFrame, locations: gpd.GeoDataFrame):
    return graph_gdf["geometry"].sindex.nearest(
        locations, return_distance=False, return_all=False
    )