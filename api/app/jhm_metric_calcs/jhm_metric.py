import geopandas as gpd
import networkx as nx
import networkit as nk
from api.app.jhm_metric_calcs import utils
from shapely.geometry import Point
from typing import Iterable, Dict

from geojson_pydantic import FeatureCollection
from collections import defaultdict
import numpy as np
from api.app.jhm_metric_calcs import constants


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
        lambda x: log_base if x < constants.MOST_COMFORTABLE_ACCS_TIME else x
    )
    gdf_houses["log_accs_time"] = (np.log10(gdf_houses[["accs_time"]])).round(2)

    gdf_houses["mean_price_rent"] = gdf_houses["mean_price_rent"].round(0)
    gdf_houses["Iq"] = (gdf_houses["mean_price_rent"] / salary).round(2)
    gdf_houses["Iq"] = (gdf_houses["Iq"] * gdf_houses["log_accs_time"]).round(2)

    gdf_houses["Iq"] = gdf_houses["Iq"].apply(
        lambda x: x if x <= 1 else constants.MAX_COEF_VALUE
    )

    gdf_houses["Iq"] = gdf_houses[["Iq", "accs_time"]].apply(
        lambda x: x["Iq"]
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
        lambda x: x.mean()
        if x["Iq"] > constants.LEAST_COMFORTABLE_IQ_COEF_VALUE
        else 0,
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
    n_cells_grid: int = 30,
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

        gdfs_results[worker.speciality]["house_points"] = Iq_coef_worker.copy()
        gdfs_results[worker.speciality]["grid_Iq"] = utils.create_grid(
            gdfs_results.get(worker.speciality).get("house_points"),
            col="Iq",
            n_cells=n_cells_grid,
        )
        gdfs_results[worker.speciality]["grid_P_avg"] = utils.create_grid(
            gdfs_results.get(worker.speciality).get("house_points"),
            col="P_avg",
            n_cells=n_cells_grid,
        )
        gdfs_results[worker.speciality]["grid_Idx"] = utils.create_grid(
            gdfs_results.get(worker.speciality).get("house_points"),
            col="Idx",
            n_cells=n_cells_grid,
        )

        # mean_Iq_coef[worker.speciality] = Iq_coef_worker["Iq"].mean()
        mask = Iq_coef_worker["Iq"] >= constants.LEAST_COMFORTABLE_IQ_COEF_VALUE
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
