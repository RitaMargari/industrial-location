import gevent.monkey
gevent.monkey.patch_all()
import grequests

from tqdm import tqdm


def dadata_create_request (org_code, token):

    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Token {token}',
    }

    json_data = {'query': str(org_code)}

    kwargs = {
        "url": 'https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/party', 
        "headers": headers, 
        "json": json_data
          }
    
    return kwargs

def dadata_decode_organization(requests):

    indices = list(range(0, len(requests), 50)) + [len(requests)]
    address = []
    city = []
    settlement = []
    region = []
    geo_lat = []
    geo_lon = []

    for left, right in tqdm(zip(indices[:-1], indices[1:]), total = len(indices)-1):
        rs = (grequests.post(**r) for r in requests[left:right])
        resp_maps = grequests.map(rs)
        
        for i in range(len(resp_maps)):
            try:
                resp = resp_maps[i].json()
                if len(resp["suggestions"]) > 0:
                    data = resp["suggestions"][0]["data"]["address"]
                    address.append(data["value"])
                    region.append(data["data"]["region"])
                    city.append(data["data"]["city"])
                    settlement.append(data["data"]["settlement"])
                    geo_lat.append(data["data"]["geo_lat"])
                    geo_lon.append(data["data"]["geo_lon"])
                else:
                    address.append(0)
                    region.append(0)
                    city.append(0)
                    settlement.append(0)
                    geo_lat.append(0)
                    geo_lon.append(0)
            except:
                address.append(0)
                region.append(0)
                city.append(0)
                settlement.append(0)
                geo_lat.append(0)
                geo_lon.append(0)

    return {
        "address": address, "region": region, "geo_lat": geo_lat, "geo_lon": geo_lon, "city": city, "settlement": settlement
        }