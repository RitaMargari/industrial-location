import gevent.monkey
gevent.monkey.patch_all()
import grequests

from tqdm import tqdm


def dadata_create_request (kladr, token):

    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Token {token}',
    }

    json_data = {'query': str(kladr)}

    kwargs = {
        "url": 'https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/address', 
        "headers": headers, 
        "json": json_data
          }
    
    return kwargs



def dadata_decode_kladr(requests):

    indices = list(range(0, len(requests), 50)) + [len(requests)]
    city = []
    region = []
    settlement = []
    area_kladr_id = []
    region_kladr_id = []
    x = []
    y = []

    for left, right in tqdm(zip(indices[:-1], indices[1:]), total = len(indices)-1):
        rs = (grequests.post(**r) for r in requests[left:right])
        resp_maps = grequests.map(rs)
        
        for i in range(len(resp_maps)):
            try:
                resp = resp_maps[i].json()
                if len(resp["suggestions"]) > 0:
                    data = resp["suggestions"][0]["data"]
                    city.append(data["city"])
                    settlement.append(data["settlement"])
                    region.append(data["region_with_type"])
                    area_kladr_id.append(data["area_kladr_id"])
                    region_kladr_id.append(data["region_kladr_id"])
                    x.append(data["geo_lat"])
                    y.append(data["geo_lon"])
                else:
                    city.append(0)
                    settlement.append(0)
                    region.append(0)
                    area_kladr_id.append(0)
                    region_kladr_id.append(0)
                    x.append(0)
                    y.append(0)
            except:
                city.append(0)
                settlement.append(0)
                region.append(0)
                area_kladr_id.append(0)
                region_kladr_id.append(0)
                x.append(0)
                y.append(0)

    return {
        'city':city, "settlement": settlement, "region": region, 
        "area_kladr_id": area_kladr_id, "region_kladr_id": region_kladr_id,
        'x': x, 'y': y
        }