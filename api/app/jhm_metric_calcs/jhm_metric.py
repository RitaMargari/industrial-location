import geopandas as gpd
import networkx as nx
import networkit as nk
from app.jhm_metric_calcs import utils
from shapely.geometry import Point

from geojson_pydantic import FeatureCollection
import numpy as np
import pyproj
import pandas as pd


def get_distance_to_work(
    G: nx.DiGraph, gdf_houses: gpd.GeoDataFrame, company_location: Point
) -> gpd.GeoDataFrame:
    """
    Graph calcs
    """
    graph_gdf, G_nx2 = utils.G_to_gdf(G)

    G_nk = utils.convert_nx2nk(G_nx2, weight="time_min")

    nodes_from_buildings = utils.get_nearest_nodes(graph_gdf, gdf_houses["geometry"])[1]
    company_node = utils.get_nearest_nodes(graph_gdf, company_location)[1]

    nk_dists = nk.distance.SPSP(G=G_nk, sources=company_node).run()
    gdf_houses["dists"] = utils.get_nk_distances(
        nk_dists=nk_dists,
        source_nodes=nodes_from_buildings,
        target_node=company_node,
    )

    return gdf_houses


def calc_coef(
    gdf_houses: gpd.GeoDataFrame, salary: int, room_area_m2: int
) -> gpd.GeoDataFrame:
    """
    Calculation of the coefficient (indicator) of job-housing
    spatial mismatch which is represented by the worker's salary,
    rent price and distance between house and workplace.

    It is also possible to spicify the flat area by room_area_m2
    so the rent price woul be calculated by average rent price per meter.
    """

    comfortable_accessibility_time = 60

    # 10 is set here because at the next step it will be transformed to 1
    # which means we do not take accessibility time into account if workplace is close to home
    log_base = 10
    gdf_houses["dists"] = gdf_houses["dists"].apply(
        lambda x: log_base if x < comfortable_accessibility_time else x
    )

    gdf_houses["log_dists"] = round(np.log10(gdf_houses[["dists"]]), 2)
    gdf_houses["calculated_rent"] = gdf_houses["avg_m2_price_rent"] * room_area_m2
    gdf_houses["calculated_rent"] = gdf_houses["calculated_rent"].round(0).astype(int)

    gdf_houses["Iq"] = gdf_houses["calculated_rent"] / salary
    gdf_houses["Iq"] = round(gdf_houses["Iq"] * gdf_houses["log_dists"], 2)

    return gdf_houses


def filter_final_coef(res: gpd.GeoDataFrame, filter: bool) -> gpd.GeoDataFrame:
    least_comfortable_coef_value = 0.7
    if filter:
        res = res[res["Iq"] <= least_comfortable_coef_value]
    return res


def fix_company_location_coords(company_location: list) -> Point:
    global_crs = 4326
    lon = company_location["lon"]
    lat = company_location["lat"]
    local_crs = utils.convert_wgs_to_utm(lon=lon, lat=lat)
    company_location = Point(pyproj.transform(global_crs, local_crs, lon, lat))
    return company_location


def calc_avg_provision(gdf_houses):
    p_columns = [col for col in gdf_houses.columns if "P_" in col]
    gdf_houses["P_avg"] = gdf_houses.loc[:, p_columns].mean(axis=1)
    return gdf_houses


def calc_avg_coef(gdf_houses):
    gdf_houses["Idx"] = gdf_houses[["P_avg", "Iq"]].mean(axis=1)
    return gdf_houses


def main(
    G: nx.DiGraph,
    gdf_houses: gpd.GeoDataFrame,
    company_location: dict,
    salary: int,
    room_area_m2: int,
    filter_coef: bool,
) -> FeatureCollection:
    company_location = fix_company_location_coords(company_location)

    res = (
        get_distance_to_work(G, gdf_houses, company_location)
        .pipe(calc_coef, salary, room_area_m2)
        .pipe(calc_avg_provision)
        .pipe(calc_avg_coef)
        .pipe(filter_final_coef, filter_coef)
    )

    return res
