import torch
import pandas as pd
import geopandas as gpd
import os
import numpy as np
from tqdm import tqdm
import shutil
from sklearn.preprocessing import OneHotEncoder

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from sklearn.preprocessing import MinMaxScaler
import warnings

tqdm.pandas()
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def pd_fill_diagonal(df_matrix, value=0):
    mat = df_matrix.values
    n = mat.shape[0]
    mat[range(n), range(n)] = value
    return pd.DataFrame(mat)



def split_train_test_new21(x, edge_index, edge_weight, mask):

    edge_index, edge_weight = edge_index, edge_weight[mask].unsqueeze(-1)

    x_s = x[edge_index[0][mask]]
    x_d = x[edge_index[1][mask]]
    x = torch.cat((x_s, x_d, edge_weight), axis=1)

    return x

# @dataclass
class DataStorer:
    DM = None
    cities = None


class MigrationDataset_2021(InMemoryDataset):
        def __init__(self, root):
            super(MigrationDataset_2021, self).__init__(root, transform=None, pre_transform=None)
            self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def raw_file_names(self):
            files = []
            # for file in os.listdir(self.root):
            #     if file.endswith(".") or file.endswith(".csv"):
            #         files.append(file)
            return files

        @property
        def processed_file_names(self):
            return ["migration_dataset"]

        def download(self):
            pass

        def process(self):
            # extract cities features
            cities = DataStorer.cities
            DM = DataStorer.DM

            # cities.set_index("region_city", inplace=True)

            # extract cities features 
            cities_features = cities[[
                "city_category", 
                "harsh_climate", 
                "ueqi_residential", 
                "ueqi_street_networks", 
                "ueqi_green_spaces", 
                "ueqi_public_and_business_infrastructure", 
                "ueqi_social_and_leisure_infrastructure",
                "ueqi_citywide_space",            
                "factories_total",
                'median_salary'
                ]]

            # encode categorical features
            one_hot = OneHotEncoder(drop='first')
            encoded_category = one_hot.fit_transform(np.expand_dims(cities["city_category"].to_numpy(), 1)).toarray()
            encoded_category_names = one_hot.get_feature_names_out(["category"])
            cities_features.loc[:, encoded_category_names] = encoded_category
            cities_features = cities_features.drop(["city_category"], axis=1)
            cities_features["harsh_climate"] = cities_features["harsh_climate"].astype(int)

                    
            cities_num = len(DM.columns)
            edge_index = [[], []]
            for i in range(cities_num):
                edge_index[0].extend([i for j in range(cities_num)])
                edge_index[1].extend([j for j in range(cities_num)])
            
            edge_index = torch.tensor(edge_index)
            # y = torch.tensor(np.concatenate(OD.to_numpy()), dtype=torch.float32)
            edge_attr = torch.tensor(np.concatenate(DM.to_numpy()), dtype=torch.float32)
            x = torch.tensor(cities_features.to_numpy(), dtype=torch.float32)
            x_names = np.array(list(cities_features.index))
            features_name = cities_features.columns
            
            # create torch object          
            graph = Data(x=x,edge_index=edge_index, edge_attr=edge_attr, x_names=x_names, features_name=features_name)
            
            data_list = []
            data_list.append(graph)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

# idx = (np.array(dt21.x_names) == city_name).nonzero()[0][0]
# mask = dt21.edge_index[0] == idx
# X = split_train_test_new21(dt21.x, dt21.edge_index, dt21.edge_attr)

def clean_preprocessed_dataset():

    try:
        print(os.getcwd()+'/processed')
        shutil.rmtree(os.getcwd()+'/app/processed')
        print('deleted')
    except Exception:
        pass 

def recalc(cities, DM, cat, migrations_all, city_name=None):

    # print('\n\n\n', cities.loc[city_name,:].iloc[4:], '\n\n\n\n')

    clean_preprocessed_dataset()
    target_vector = None
    
    root="app"
    DataStorer.cities = cities
    DataStorer.DM = DM
    
    dtst = MigrationDataset_2021(root)
    dt21 = dtst.data

    if city_name:
        migrations_all = pd.DataFrame()

        for name in tqdm(dt21.x_names):
            idx = (np.array(dt21.x_names) == name).nonzero()[0][0]
            mask = dt21.edge_index[0] == idx
            X = split_train_test_new21(dt21.x, dt21.edge_index, dt21.edge_attr, mask)

            # cat = CatBoostRegressor().load_model(
            #     "/home/gk/vscode/RITA_CATBOOST/cat_model_dummies_28"
            # )
            res = cat.predict(X.numpy())
            if name == city_name:
                print(res.sum())
                target_vector = X
            migrations_all[name] = res

        # migrations_all = pd_fill_diagonal(migrations_all)
        # migrations_all[migrations_all < 0] = 0
        migrations_all.columns = cities.index
        migrations_all.index = migrations_all.columns
        

    else:
        migrations_all = migrations_all

    

    migrations_to_each_city = migrations_all.sum(axis=0)
    migrations_from_each_city = migrations_all.sum(axis=1)

    
    if city_name:
        print('\n\n\n',migrations_all.loc[:, city_name].sum())
        print(migrations_all.loc[city_name,:].sum(), '\n\n\n')
        migrations_to_selected_city = migrations_all.loc[:, city_name]
        cities.loc[:, "migrations_to_selected_city"] = migrations_to_selected_city
        cities.loc[cities["migrations_to_selected_city"] < 0, "migrations_to_selected_city"] = 0
        # migrations_from_selected_city = migrations_all.loc[city_name, :]

    responses_predict = pd.DataFrame(
        {
            "cluster_center_cv": [i for i in cities.index],
            "cluster_center_vacancy": city_name,
        }
    )
    responses_predict["responses"] = migrations_to_each_city.values

    cities.loc[:, "migrations_to_each_city"] = migrations_to_each_city.values

    cities.loc[:, "migrations_from_each_city"] = migrations_from_each_city.values
    cities.loc[:, "probability_to_move"] = (
        cities.loc[:, "migrations_from_each_city"]
        / cities.loc[:, "migrations_to_each_city"]
    )
    cities.loc[cities["probability_to_move"] > 1_000_000, "probability_to_move"] = 0
    cities.loc[cities["probability_to_move"] < -1_000_000, "probability_to_move"] = 0

    cities.loc[cities["vacancy_count"] < 1, "vacancy_count"] = 1
    num_vacancy_each_city = cities[
        "vacancy_count"
    ]  # the total number of relevant vacancies in a city
    
    cities_update = cities.copy()
    cities_update.loc[:, "num_in_migration"] = migrations_to_each_city.values
    cities_update.loc[:, "one_vacancy_out_response"] = (
        cities_update["migrations_to_each_city"] / num_vacancy_each_city.values
    ).round(3)

    cities_update.loc[
        cities_update["one_vacancy_out_response"] > 1_000_000, "one_vacancy_out_response"
    ] = 0
    cities_update.loc[
        cities_update["one_vacancy_out_response"] < -1_000_000, "one_vacancy_out_response"
    ] = 0

    scaler = MinMaxScaler()

    # cities_update['vac_to_fac'] = cities_update['factories_total'] / cities_update['vacancy_count']
    column_norm = [
        "probability_to_move",
        "one_vacancy_out_response",
        "factories_total",
        "vacancy_count",
    ]

    for column in ["cv_count_weighted_sum", "graduates_weighted_sum"]:
        if column in cities_update.columns:
            column_norm.append(column)

    migration_estinate = pd.DataFrame(
        scaler.fit_transform(np.log(cities_update[column_norm].abs().to_numpy() + 10e-06)),
        index=cities_update.index,
        columns=column_norm,
    )

    cities_update["estimate"] = 0


    for column in column_norm:
        cities_update["estimate"] += migration_estinate[column]

    scaler = MinMaxScaler()
    cities_update["estimate"] = (
        pd.Series(
            scaler.fit_transform(
                np.expand_dims(cities_update["estimate"].to_numpy(), 1)
            ).squeeze(),
            index=cities_update.index,
        )
        .fillna(0)
        .round(3)
    )

    if city_name:
        cities_update = cities_update.rename(
            columns={
                "estimate": "estimate_after",
                "num_in_migration": "num_in_migration_after",
            }
        )
    clean_preprocessed_dataset()

    return cities_update, target_vector
        # pd.set_option("display.max_columns", None)

        # city_update[["estimate_after", "num_in_migration_after"]]
