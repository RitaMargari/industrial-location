from pandas.core.frame import DataFrame
import pandas as pd
import geopandas as gpd
import networkx as nx

from typing import Optional, List


class NetworkConstruction:
    def __init__():
        pass

    def filter_responses(responses):
        responses = responses[
            [
                "date_creation",
                "id_candidate",
                "id_cv",
                "id_hiring_organization",
                "id_response",
                "id_vacancy",
                "region_code",
                "response_type",
            ]
        ]

        return responses

    def filter_responses_by_industry(responses, cv, specialists, ontology, industry):
        if industry:
            print('industry')
            responses = responses[
                responses["id_cv"].isin(
                    cv[
                        cv["id_candidate"].isin(
                            specialists[
                                specialists["label"].isin(
                                    ontology[ontology["Отрасль"].isin(industry)][
                                        "Специальность"
                                    ]
                                )
                            ]["id_candidate"]
                        )
                    ]["id_cv"]
                )
            ]

        return responses

    def filter_responses_by_speciality(responses, cv, specialists, group):
        if group:
            responses = responses[
                responses["id_cv"].isin(
                    cv[
                        cv["id_candidate"].isin(
                            specialists[specialists["label"].isin(group)][
                                "id_candidate"
                            ]
                        )
                    ]["id_cv"]
                )
            ]

        return responses

    def get_response_year(responses):
        responses.loc[:, "year"] = responses["date_creation"].apply(
            lambda x: int(float(x.split("-")[0]))
        )
        return responses

    def rename_vacancies_coordinates(vacancies):
        vacancies = vacancies.rename(
            columns={
                "job_location_geo_latitude": "x",
                "job_location_geo_longitude": "y",
            }
        )
        return vacancies

    def filter_kladr(kladr_2021):
        kladr_2021 = kladr_2021[
            (kladr_2021["region_name"] != "Москва")
            & (kladr_2021["region_name"] != "Санкт-Петербург")
            & (kladr_2021["region_name"] != "Севастополь")
        ]

        return kladr_2021

    def filter_cv_years(cv_years, kladr_2021, cv_cities):
        regions = list(kladr_2021["region_name"].unique())
        cv_cities = cv_cities[~cv_cities["city"].isin(regions)]

        cv_years = cv_years.join(
            cv_cities.set_index("kladr")[["city", "x", "y"]], on="locality"
        )
        cv_years = cv_years[cv_years["city"].notna()]

        return cv_years

    def merge_responses_cv_vacancy(responses, cv_years, vacancies):
        responses = responses.merge(
            cv_years[["id_cv", "year", "city", "x", "y"]],
            left_on=["id_cv", "year"],
            right_on=["id_cv", "year"],
        )

        responses = responses.join(
            vacancies.set_index("identifier")[["city", "x", "y"]],
            on="id_vacancy",
            lsuffix="_cv",
            rsuffix="_vacancy",
        )
        return responses

    def filter_organisations(organizations, responses):
        organizations["org_code"] = organizations["inn"].fillna(organizations["ogrn"])
        organization_list = responses[responses["city_vacancy"].isna()][
            "id_hiring_organization"
        ].unique()
        org_codes = list(
            organizations[organizations["id_organization"].isin(organization_list)][
                "org_code"
            ].astype(int)
        )

        return org_codes

    def filter_companies_info(organizations, org_cities, org_codes):
        org_cities["org_code"] = org_codes
        org_cities["data.address.data.city"] = org_cities[
            "data.address.data.city"
        ].fillna(org_cities["data.address.data.settlement"])
        org_cities = org_cities[org_cities["data.address.data.city"] != 0]
        org_cities = org_cities.join(
            organizations.set_index("org_code")["id_organization"],
            on="org_code",
            how="inner",
        )

        org_cities = org_cities[
            [
                "org_code",
                "id_organization",
                "data.address.data.geo_lat",
                "data.address.data.geo_lon",
                "data.address.data.city",
            ]
        ]

        org_cities.rename(
            columns={
                "data.address.data.geo_lat": "geo_lat",
                "data.address.data.geo_lon": "geo_lon",
                "data.address.data.city": "city",
            },
            inplace=True,
        )

        return org_cities

    def merge_responses_org_cities(responses, org_cities):
        responses = responses.join(
            org_cities.set_index("id_organization")[["city", "geo_lat", "geo_lon"]],
            on="id_hiring_organization",
        )
        responses["city_vacancy"] = responses["city_vacancy"].fillna(responses["city"])
        responses["x_vacancy"] = responses["x_vacancy"].fillna(responses["geo_lat"])
        responses["y_vacancy"] = responses["y_vacancy"].fillna(responses["geo_lon"])

        return responses

    def filter_responses_by_years(responses, years):
        if years:
            responses = responses[responses["year"].isin(years)]
        return responses

    def drop_responses_without_city(responses):
        responses = responses.dropna(subset=["city_vacancy"])

        return responses

    def create_cities_gdf(responses):
        cities_gdf = (
            pd.concat(
                [
                    responses[["city_cv", "x_cv", "y_cv", "year"]].rename(
                        columns={"city_cv": "city", "x_cv": "x", "y_cv": "y"}
                    ),
                    responses[["city_vacancy", "x_vacancy", "y_vacancy"]].rename(
                        columns={
                            "city_vacancy": "city",
                            "x_vacancy": "x",
                            "y_vacancy": "y",
                        }
                    ),
                ]
            )
            .drop_duplicates(subset="city")
            .reset_index(drop=True)
        )

        cities_gdf = gpd.GeoDataFrame(
            cities_gdf, geometry=gpd.points_from_xy(cities_gdf["y"], cities_gdf["x"])
        )

        cities_gdf = cities_gdf[~cities_gdf.is_empty].reset_index(drop=True)

        return cities_gdf

    def create_edges(responses, cities_gdf):
        edges = (
            responses.groupby(["city_cv", "city_vacancy"])
            .size()
            .reset_index(name="weight")
        )

        edges = edges[edges["city_cv"].isin(cities_gdf["city"].values)]
        edges = edges[edges["city_vacancy"].isin(cities_gdf["city"].values)]
        edges.reset_index(drop=True, inplace=False)

        return edges

    def create_G(cities_gdf, edges):
        G = nx.DiGraph()
        G.add_nodes_from(cities_gdf["city"].values)

        for row in edges.itertuples():
            G.add_edge(row.city_cv, row.city_vacancy, weight=row.weight)

        for node in G.nodes():
            if node in cities_gdf["city"].values:
                G.nodes[node]["geometry"] = str(
                    cities_gdf.loc[cities_gdf["city"] == node, "geometry"].item()
                )

        return G

    def run_pipeline(
        responses=None,
        vacancies=None,
        kladr_2021=None,
        organizations=None,
        dadata_organisations_info=None,
        cv_years=None,
        cv_cities=None,
        cv: Optional[DataFrame] = None,
        ontology: Optional[DataFrame] = None,
        specialists: Optional[DataFrame] = None,
        group: Optional[List] = None,
        industries: Optional[List] = None,
        years: Optional[List] = None,
        pre_calculated_responses: DataFrame = None,
        return_responses=False,
    ):
        if not isinstance(pre_calculated_responses, DataFrame):
            vacancies = vacancies.pipe(NetworkConstruction.rename_vacancies_coordinates)
            kladr_2021 = kladr_2021.pipe(NetworkConstruction.filter_kladr)
            cv_years = cv_years.pipe(
                NetworkConstruction.filter_cv_years, kladr_2021, cv_cities
            )

            responses = (
                responses.pipe(NetworkConstruction.filter_responses)
                .pipe(NetworkConstruction.get_response_year)
                .pipe(
                    NetworkConstruction.merge_responses_cv_vacancy, cv_years, vacancies
                )
            )

            organisation_ids = NetworkConstruction.filter_organisations(
                organizations, responses
            )

            organizations = organizations.pipe(
                NetworkConstruction.filter_companies_info,
                dadata_organisations_info,
                organisation_ids,
            )

            responses = responses.pipe(
                NetworkConstruction.merge_responses_org_cities, organizations
            ).pipe(NetworkConstruction.drop_responses_without_city)

        else:
            responses = pre_calculated_responses

        responses = (
            responses.pipe(NetworkConstruction.filter_responses_by_years, years)
            .pipe(
                NetworkConstruction.filter_responses_by_industry,
                cv,
                specialists,
                ontology,
                industries,
            )
            .pipe(
                NetworkConstruction.filter_responses_by_speciality,
                cv,
                specialists,
                group,
            )
        )

        cities_gdf = responses.pipe(NetworkConstruction.create_cities_gdf)
        edges = responses.pipe(NetworkConstruction.create_edges, cities_gdf)
        G = cities_gdf.pipe(NetworkConstruction.create_G, edges)

        if return_responses:
            return G, responses
        else:
            return G
