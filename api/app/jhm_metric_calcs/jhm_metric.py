import geopandas as gpd
import networkx as nx
import networkit as nk
from app.jhm_metric_calcs import utils
from shapely.geometry import Point

from geojson_pydantic import FeatureCollection
from collections import defaultdict
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
    gdf_houses["accs_time"] = utils.get_nk_distances(
        nk_dists=nk_dists,
        source_nodes=nodes_from_buildings,
        target_node=company_node,
    )

    return gdf_houses


def calc_coef(gdf_houses: gpd.GeoDataFrame, salary: int) -> gpd.GeoDataFrame:
    """
    Calculation of the coefficient (indicator) of job-housing
    spatial mismatch which is represented by the worker's salary,
    rent price and distance between house and workplace.

    It is also possible to spicify the flat area by room_area_m2
    so the rent price woul be calculated by average rent price per meter.
    """

    MOST_COMFORTABLE_ACCS_TIME = 15
    LEAST_COMFORTABLE_ACCS_TIME = 90

    # 10 is set here because at the next step it will be transformed to 1
    # which means we do not take accessibility time into account if workplace is close to home
    log_base = 10
    gdf_houses["accs_time"] = gdf_houses["accs_time"].apply(
        lambda x: log_base if x < MOST_COMFORTABLE_ACCS_TIME else x
    )

    gdf_houses["log_accs_time"] = (np.log10(gdf_houses[["accs_time"]])).round(2)
    gdf_houses["mean_price_rent"] = gdf_houses["mean_price_rent"].round(0)
    gdf_houses["Iq"] = (gdf_houses["mean_price_rent"] / salary).round(2)
    gdf_houses["Iq"] = (gdf_houses["Iq"] * gdf_houses["log_accs_time"]).round(2)
    gdf_houses["Iq"] = gdf_houses["Iq"].apply(lambda x: x if x <= 1 else 1)

    gdf_houses["Iq"] = gdf_houses[["Iq", "accs_time"]].apply(
        lambda x: x["Iq"] if x["accs_time"] <= LEAST_COMFORTABLE_ACCS_TIME else 1,
        axis=1,
    )

    return gdf_houses


def fix_company_location_coords(company_location: dict) -> Point:
    global_crs = 4326
    lon = company_location["lon"]
    lat = company_location["lat"]
    company_location = Point([lon, lat])
    local_crs = utils.convert_wgs_to_utm(lon=company_location.y, lat=company_location.x)
    company_location = gpd.GeoSeries(Point(company_location.y, company_location.x), crs=global_crs).to_crs(local_crs)

    return company_location


def calc_avg_provision(gdf_houses):
    p_columns = [col for col in gdf_houses.columns if "P_" in col]
    gdf_houses["P_avg"] = gdf_houses.loc[:, p_columns].mean(axis=1)
    return gdf_houses


def calc_avg_coef(gdf_houses):
    gdf_houses["Iq"] = 1 - gdf_houses["Iq"]
    gdf_houses["Idx"] = gdf_houses[["P_avg", "Iq"]].apply(
        lambda x: x.mean() if x["Iq"] > 0.3 else 0, axis=1
    )
    return gdf_houses


def calc_jhm_main(
    G: nx.DiGraph,
    gdf_houses: gpd.GeoDataFrame,
    company_location: dict,
    salary: int,
) -> FeatureCollection:
    company_location = fix_company_location_coords(company_location)

    res = (
        get_distance_to_work(G, gdf_houses, company_location)
        .pipe(calc_coef, salary)
        .pipe(calc_avg_provision)
        .pipe(calc_avg_coef)
    )

    res["geometry"] = res["geometry"].representative_point()

    return res


def main(gdf_houses, worker_and_salary, graph, company_location):
    LEAST_COMFORTABLE_IQ_COEF_VALUE = 0.3

    gdfs_results = defaultdict(
        gpd.GeoDataFrame
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

        gdfs_results[worker.speciality] = Iq_coef_worker.copy()
        gdfs_results[worker.speciality] = utils.create_grid(gdfs_results.get(worker.speciality))

        # mean_Iq_coef[worker.speciality] = Iq_coef_worker["Iq"].mean()
        mask = Iq_coef_worker["Iq"] >= LEAST_COMFORTABLE_IQ_COEF_VALUE
        Iq_coef_worker_tmp = Iq_coef_worker[mask].copy()

        for col in provision_columns:
            P_mean_val = Iq_coef_worker_tmp.loc[:, col].mean()
            K1[worker.speciality][f"{col}_avg"] = round(P_mean_val, 2)
            K3[worker.speciality][f"{col}_K"] = (
                K1[worker.speciality][f"{col}_avg"] / K2[f"{col}_avg_all_houses"]
            )

        K4[f"{worker.speciality}_avg"] = np.mean(
            [val for inner_dict in K3.values() for val in inner_dict.values()]
        )

    all_K = [val for val in K4.values()]
    K = np.mean(all_K)
    D = np.max(all_K) / np.median(all_K)

    conditions = [(K >= 1), (K > 0.7) & (K < 1), (K <= 0.7)]
    values = ["green", "orange", "red"]

    K_color = np.select(conditions, values)

    print(
        "\n\n K:",
        K,
        "\n\n D:",
        D,
        "\n\n K_color:",
        K_color,
        "\n",
    )

    return {"K": K, "D": D, "K_color": K_color, "gdfs": gdfs_results}
