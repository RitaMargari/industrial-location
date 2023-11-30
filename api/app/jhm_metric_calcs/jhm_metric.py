import geopandas as gpd
import networkx as nx
import networkit as nk
from ..jhm_metric_calcs import utils
from shapely.geometry import Point
from typing import Iterable, Dict
import json

from geojson_pydantic import FeatureCollection
from collections import defaultdict
import numpy as np
from  app import constants


def calc_accs_via_spsp(
    G_nk, company_node: gpd.GeoSeries, nodes_from_buildings: gpd.GeoSeries
):
    nk_dists = nk.distance.SPSP(G=G_nk, sources=company_node).run()
    return utils.get_nk_distances(
        nk_dists=nk_dists,
        source_nodes=nodes_from_buildings,
        target_node=company_node,
    )


def get_accs_time_to_work(
    G: nx.DiGraph, gdf_houses: gpd.GeoDataFrame, company_location: gpd.GeoSeries
) -> gpd.GeoDataFrame:
    """
    Graph calcs
    """
    graph_gdf, G_nx2 = utils.G_to_gdf(G)
    G_nk = utils.convert_nx2nk(G_nx2, weight="time_min")
    nodes_from_buildings = utils.get_nearest_nodes(graph_gdf, gdf_houses["geometry"])[1]
    company_node = utils.get_nearest_nodes(graph_gdf, company_location)[1]
    gdf_houses["accs_time"] = calc_accs_via_spsp(
        G_nk, company_node, nodes_from_buildings
    )

    return gdf_houses


def calc_iq_coef(gdf_houses: gpd.GeoDataFrame, salary: int) -> gpd.GeoDataFrame:
    """
    Calculation of the coefficient (indicator) of job-housing
    spatial mismatch which is represented by the worker's salary,
    rent price and distance between house and workplace.

    It is also possible to spicify the flat area by room_area_m2
    so the rent price woul be calculated by average rent price per meter.
    """

    # 10 is set here because at the next step it will be transformed to 1
    # which means we do not take accessibility time into account if workplace is close to home
    log_base = 10
    gdf_houses["accs_time"] = gdf_houses["accs_time"].apply(
        lambda x: log_base if x < constants.MOST_COMFORTABLE_ACCS_TIME else round(x, 3)
    )

    
    gdf_houses["log_accs_time"] = (np.log10(gdf_houses[["accs_time"]])).round(3)

    gdf_houses["price"] = gdf_houses["price"].round(0)
    gdf_houses["Iq"] = (gdf_houses["price"] / salary).round(3)
    gdf_houses["Iq"] = (gdf_houses["Iq"] * gdf_houses["log_accs_time"]).round(3)

    gdf_houses["Iq"] = gdf_houses["Iq"].apply(
        lambda x: round(x, 3) if x <= 1 else constants.MAX_COEF_VALUE
    )

    gdf_houses["Iq"] = gdf_houses[["Iq", "accs_time"]].apply(
        lambda x: round(x["Iq"], 3)
        if x["accs_time"] <= constants.LEAST_COMFORTABLE_ACCS_TIME
        else constants.MAX_COEF_VALUE,
        axis=1,
    )

    return gdf_houses


def convert_wgs_point_to_utm_geoseries(company_location_dict: dict) -> gpd.GeoSeries:
    local_crs = utils.convert_wgs_to_utm(
        lon=company_location_dict["lat"], lat=company_location_dict["lon"]
    )
    company_location = gpd.GeoSeries(
        Point(company_location_dict["lat"], company_location_dict["lon"]),
        crs=constants.CRS_WGS84,
    ).to_crs(local_crs)

    return company_location


def calc_avg_provision(gdf_houses: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    p_columns = [col for col in gdf_houses.columns if "P_" in col]
    gdf_houses["P_avg"] = gdf_houses.loc[:, p_columns].mean(axis=1)
    return gdf_houses


def invert_iq_coef(gdf_with_iq_coef: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf_with_iq_coef["Iq"] = 1 - gdf_with_iq_coef["Iq"]
    return gdf_with_iq_coef


def calc_avg_coef(gdf_houses: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf_houses["Idx"] = gdf_houses[["P_avg", "Iq"]].apply(
        lambda x: 0
        if x["Iq"] < constants.LEAST_COMFORTABLE_IQ_COEF_VALUE \
        or x["P_avg"] < constants.LEAST_COMFORTABLE_P_AVG_VALUE \
        else x.mean(),
        axis=1,
    )
    return gdf_houses


def calc_jhm_main(
    G: nx.DiGraph,
    gdf_houses: gpd.GeoDataFrame,
    company_location: Dict[str, float],
    salary: int,
) -> FeatureCollection:
    company_location = convert_wgs_point_to_utm_geoseries(company_location)

    res = (
        get_accs_time_to_work(G, gdf_houses, company_location)
        .pipe(calc_iq_coef, salary)
        .pipe(calc_avg_provision)
        .pipe(invert_iq_coef)
        .pipe(calc_avg_coef)
    )

    res["geometry"] = res["geometry"].representative_point()

    return res


def main(
    gdf_houses: gpd.GeoDataFrame,
    worker_and_salary: Iterable[Dict[str, float]],
    graph: nx.DiGraph,
    company_location: Dict[str, float],
    cell_size_meters: int,
):
    gdfs_results = defaultdict(
        lambda: defaultdict(gpd.GeoDataFrame)
    )  # gdf with calculated coef for each house for each specified worker
    # mean_Iq_coef = defaultdict(float)  #
    K1 = defaultdict(
        lambda: defaultdict(float)
    )  # avg P_{provision_service} in the nearest comfortable area
    K2 = defaultdict(float)  # avg P_{provision_service}

    K3 = defaultdict(
        lambda: defaultdict(float)
    )  # avg estimation on total workers' comfortability regarding to the enterprise location
    K4 = defaultdict(float)

    provision_columns = [col for col in gdf_houses.columns if "P_" in col]
    for col in provision_columns:
        K2[f"{col}_avg_all_houses"] = round(gdf_houses.loc[:, col].mean(), 2)

    for worker in worker_and_salary:
        Iq_coef_worker = calc_jhm_main(
            G=graph,
            gdf_houses=gdf_houses,
            company_location=company_location,
            salary=worker.salary,
        )

        if Iq_coef_worker.shape[0] > 1:
            gdfs_results[worker.speciality]["house_points"] = Iq_coef_worker.copy()

            coefs = ["Iq", "P_avg", "Idx"]
            gdfs_results[worker.speciality]["grid"] = utils.create_grid(
                gdfs_results.get(worker.speciality).get("house_points"),
                cols=coefs,
                cell_size_meters=cell_size_meters,
            )
            gdfs_results[worker.speciality]["grid"].to_crs(constants.CRS_WGS84, inplace=True)
            gdfs_results[worker.speciality]["grid"] = gdfs_results[worker.speciality]["grid"].to_json()
            

            # filter gdf by jobs-housing match value
            mask = Iq_coef_worker["Iq"] >= constants.LEAST_COMFORTABLE_IQ_COEF_VALUE
            Iq_coef_worker = Iq_coef_worker[mask].copy()

            if Iq_coef_worker.shape[0] == 0:
                K4[f"{worker.speciality}_avg"] = 0
                continue
            
            gdfs_results[worker.speciality]["house_points"] = Iq_coef_worker.copy()
            gdfs_results[worker.speciality]["house_points"].to_crs(
                constants.CRS_WGS84, inplace=True
            )
            gdfs_results[worker.speciality]["house_points"] = gdfs_results[worker.speciality]["house_points"].to_json()
            

            for col in provision_columns:
                P_mean_val = Iq_coef_worker.loc[:, col].mean()
                K1[worker.speciality][f"{col}_avg"] = round(P_mean_val, 2)
                K3[worker.speciality][f"{col}_K"] = (
                    K1[worker.speciality][f"{col}_avg"] / K2[f"{col}_avg_all_houses"]
                )

            K4[f"{worker.speciality}_avg"] = round(
                np.mean(
                    [val for inner_dict in K3.values() for val in inner_dict.values()]
                ),
                3,
            )

        else:
            # 0.01 is the least possible value
            # it indicates that there are no suitable houses to live comfortably
            for col in provision_columns:
                K3[worker.speciality][f"{col}_K"] = 0.01
            K4[f"{worker.speciality}_avg"] = 0.01

    all_K = [val for val in K4.values()]
    K = round(np.mean(all_K), 3)
    D = round(np.max(all_K) - np.median(all_K), 3)

    print(
        "\n\n K:",
        all_K,
        "\n\n D:",
        D,
        '\n', K4,
    )

    return {"K": K, "D": D, "K4": K4, "gdfs": gdfs_results}
