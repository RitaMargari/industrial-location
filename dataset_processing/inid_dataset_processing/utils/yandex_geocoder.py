import gevent.monkey
gevent.monkey.patch_all()
import grequests

from tqdm import tqdm

def yandex_create_urls (adr, api_key):
    base_request = f'https://geocode-maps.yandex.ru/1.x/?apikey={api_key}&format=json&geocode={adr}'
    return base_request

def yandex_geocode(urls):
    indices = list(range(0, len(urls), 100)) + [len(urls)]
    x = []
    y = []
    yand_adr = []
    locality = []
    name = []
    for left, right in tqdm(zip(indices[:-1], indices[1:]), total = len(indices)-1):
        rs = (grequests.get(urls) for urls in urls[left:right])
        resp_maps = grequests.map(rs)
        for i in range(len(resp_maps)):
            try:
                resp = resp_maps[i].json()
                members = len(resp['response']['GeoObjectCollection']['featureMember'])
                if members > 0:
                    geoobj = resp['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']
                    y.append(float(geoobj['Point']['pos'].split()[0]))
                    x.append(float(geoobj['Point']['pos'].split()[1]))
                    yand_adr.append(geoobj['metaDataProperty']['GeocoderMetaData']['text'])
                    name.append(geoobj["name"])
                    try:
                        adr = geoobj['metaDataProperty']['GeocoderMetaData']["AddressDetails"]["Country"]["AdministrativeArea"]
                        if "Locality" in adr:
                            locality.append(adr["Locality"]["LocalityName"])
                        elif "SubAdministrativeArea" in adr:
                            if "Locality" in adr["SubAdministrativeArea"]:
                                if "LocalityName" in adr["SubAdministrativeArea"]["Locality"]:
                                    locality.append(adr["SubAdministrativeArea"]["Locality"]["LocalityName"])
                                else:
                                    locality.append(0)
                            else:
                                locality.append(adr["SubAdministrativeArea"]["SubAdministrativeAreaName"])
                        else:
                            locality.append(0)
                    except:
                        locality.append(0)
                else:
                    x.append(0)
                    y.append(0) 
                    yand_adr.append(0)
                    locality.append(0)
                    name.append(0)
            except:
                geoobj = resp['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']
                print(
                    "Something wierd in response structure for coordinates:", 
                    float(geoobj['Point']['pos'].split()[0]),
                    float(geoobj['Point']['pos'].split()[1])
                    )
                x.append(0)
                y.append(0) 
                yand_adr.append(0)
                locality.append(0)
                name.append(0)
                
    return {'x':x,'y':y,'yand_adr':yand_adr, 'locality':locality, "name": name}