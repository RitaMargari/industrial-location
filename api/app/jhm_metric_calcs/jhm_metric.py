import geopandas as gpd
import networkx as nx
import networkit as nk
from app.jhm_metric_calcs import utils
from shapely.geometry import Point
import numpy as np
import json
import pandas as pd


class JhmMetric:
    def get_distance_to_work(
        G: nx.DiGraph,
        house_prices: gpd.GeoDataFrame,
        company_location: Point,
        local_crs: int,
    ) -> gpd.FeatureCollection:
        """
        Graph calcs
        """
        graph_gdf, G_nx2 = utils.G_to_gdf(G, local_crs)

        G_nk = utils.convert_nx2nk(G_nx2, weight="time_min")

        nodes_from_buildings = utils.get_nearest_nodes(
            graph_gdf, house_prices["geometry"]
        )[1]
        company_node = utils.get_nearest_nodes(graph_gdf, company_location)[1]

        nk_dists = nk.distance.SPSP(G=G_nk, sources=company_node).run()
        house_prices["dists"] = utils.get_nk_distances(
            nk_dists=nk_dists,
            source_nodes=nodes_from_buildings,
            target_node=company_node,
        )

        # bug fix since sometimes idk why there are nodes that are not connected with the graph
        max_time_in_city = 400
        house_prices.loc[
            house_prices["dists"] > max_time_in_city, "dists"
        ] = house_prices["dists"].median()

        return house_prices

    def calc_coef(
        house_prices: gpd.GeoDataFrame, salary: int, room_area_m2: int
    ) -> gpd.GeoDataFrame:
        """
        Calculation of the coefficient (indicator) of job-housing
        spatial mismatch which is represented by the worker's salary,
        rent price and distance between house and workplace.

        It is also possible to spicify the flat area by room_area_m2
        so the rent price woul be calculated by average rent price per meter.
        """

        comfortable_accessibility_time = 15

        # 10 is set here because at the next step it will be transformed to 1
        # which means we do not take accessibility time into account if workplace is close to home
        house_prices["dists"] = house_prices["dists"].apply(
            lambda x: 10 if x < comfortable_accessibility_time else x
        )

        house_prices["coef"] = house_prices["avg_m2_price_rent"] * room_area_m2 / salary
        house_prices["log_dists"] = np.log10(house_prices[["dists"]])
        house_prices["coef"] = house_prices["coef"] * house_prices["log_dists"]
        house_prices = house_prices.loc[
            :, ["coef", "avg_m2_price_rent", "log_dists", "geometry"]
        ]
        return house_prices

    def filter_coef(res: gpd.GeoDataFrame, filter: bool) -> gpd.GeoDataFrame:
        least_comfortable_coef_value = 0.7
        if filter:
            res = res[res["coef"] <= least_comfortable_coef_value]
        return res

    def main(
        G: nx.DiGraph,
        house_prices: gpd.GeoDataFrame,
        company_location: Point,
        salary: int,
        room_area_m2: int,
        filter_coef=True,
        return_json=True,
        local_crs=32636,
        debug_mode=False,
    ):
        res = (
            JhmMetric.get_distance_to_work(G, house_prices, company_location, local_crs)
            .pipe(JhmMetric.calc_coef, salary, room_area_m2)
            .pipe(JhmMetric.filter_coef, filter_coef)
        )

        if return_json:
            return json.loads(res.to_json())

        elif debug_mode:
            # debug mode return histogram values since swager ui cant load whole geojson as an output
            bins = 40
            out = pd.cut(res["coef"], bins=bins, ordered=True).value_counts().to_dict()
            out = {str(key): value for key, value in out.items()}
            return out

        else:
            return res
