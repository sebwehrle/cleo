# %% imports
import os
import subprocess
import logging
import pandas as pd
import geopandas as gpd

import json
import numpy as np
from pathlib import Path
from cleo.utils import download_file
from urllib.request import urlretrieve
from shapely.geometry import Point

# TODO: gdal-bin needs to be installed for processing GIP
# sudo apt update
# sudo apt install gdal-bin


def process_gip(gip_path):
    # documented in https://www.gip.gv.at/assets/downloads/GIP_Standardbeschreibung_2.3.2_FINAL.pdf
    # street categories are explained in 2.3.2.6 Abschnittskategorie (GIP.EDGE.EDGECATEGORY)
    # convert to shapefiles
    os.chdir(gip_path)
    subprocess.run(
        """ogr2ogr -f "ESRI shapefile" shp gip_network_ogd.gpkg -sql "select cast(EDGECAT as character(255)) from EDGE_OGD" -overwrite -dialect ogrsql""",
        shell=True)

    # read data
    gip_shp = gip_path / 'shp' / 'EDGE_OGD.shp'

    # abbreviations:
    # A: Autobahn / highway
    # S: Schnellstraße / highway
    # B: Landesstraße B (ehem. Bundesstraße)
    # L: Landesstraße L (ehem. Landeshauptstraße)
    hochrangig = ['A', 'S', 'B', 'L']
    hrng = gpd.GeoDataFrame()

    # total number of rows: 1 532 485
    for n in range(1, 63):
        gip = gpd.read_file(gip_shp, rows=slice((n - 1) * 25000, n * 25000))
        gip = gip.loc[gip['EDGECAT'].isin(hochrangig)]
        hrng = pd.concat([hrng, gip])

    hrng = hrng.dissolve()
    # hrng.to_file(ROOTDIR / 'data/gip/hrng_streets.shp')
    logging.info('High-level road grid preprocessed')
    return hrng


def process_water_bodies(water_bodies_path):
    # process water bodies
    main_water_bodies = ['100 km² Gewässer', '1000 km² Gewässer', '10000 km² Gewässer', '500 km² Gewässer',
                         '4000 km² Gewässer']
    running = gpd.read_file(water_bodies_path / 'Fliessgewaesser.shp')
    running = running.loc[running['GEW_KAT'].isin(main_water_bodies)]
    running = running.dissolve()
    # wbd.to_file(ROOTDIR / 'data/water_bodies/main_running_waters.shp')

    standing = gpd.read_file(water_bodies_path / 'StehendeGewaesser.shp')
    standing = standing.loc[standing['FLAECHEKM2'] >= 0.03125, :]
    standing = standing.dissolve()
    # lks.to_file(ROOTDIR / 'data/water_bodies/main_standing_waters.shp')

    logging.info('Water bodies preprocessed')
    return running, standing


def process_tourism(data_path):

    # read geodata on austrian municipalities
    austria = gpd.read_file(data_path / 'vgd' / 'vgd_oesterreich.shp')
    austria = austria[['GKZ', 'PG', 'BL', 'geometry']].dissolve(by='GKZ')
    austria.reset_index(inplace=True)
    austria['GKZ'] = austria['GKZ'].astype('int')

    # process tourism data
    stays = pd.read_excel(data_path / 'tourism' / 'Tabelle 30 GEH 2018.xlsx',
                          header=[0, 1, 2, 3], skiprows=[0, 5, 6], na_values=['GEH'])
    stays = stays.loc[~stays.iloc[:, 0].isna(), :]
    stays = stays.loc[[not isinstance(e, str) for e in stays.iloc[:, 0]], :]

    col = pd.DataFrame.from_records(data=stays.columns.to_flat_index())
    col = col.replace({"-\n": "", "\n": " ", " ": ""}, regex=True)
    for i in range(0, 4):
        col.loc[col[i].str.contains('Unnamed'), i] = ''

    col = [col.loc[row, :].str.cat(sep=' ').strip() for row in col.index]

    winter = 'WINTERSAISON2017/2018 ÜBERNACHTUNGEN INSGESAMT'
    summer = 'SOMMERSAISON2018 ÜBERNACHTUNGEN INSGESAMT'

    stays.columns = col
    stays['GEM.KENNZIFFER'] = stays['GEM.KENNZIFFER'].astype('int')
    stays[winter] = stays[[winter]].replace('-', 0).astype('float')
    stays[summer] = stays[[summer]].astype('float').fillna(0)
    stays['overnight_stays'] = stays[[winter, summer]].sum(axis=1)

    # merge overnight stays into municipality-geodata
    geostays = austria.merge(stays[['GEM.KENNZIFFER', 'overnight_stays']], left_on='GKZ', right_on='GEM.KENNZIFFER',
                             how='left')
    geostays['overnight_stays'] = geostays['overnight_stays'].fillna(0)
    # geostays.to_file(ROOTDIR / 'data/tourism/overnight_stays.shp')

    logging.info('Touristic overnight stays preprocessed')
    return geostays


def process_windturbines():

    # Settings for downloading and processing wind turbine data from IG Windkraft
    igwurl = 'https://www.igwindkraft.at/src_project/external/maps/generated/gmaps_daten.js'
    data_dir = Path('ROOTDIR') / 'data' / 'AT_turbines'
    igw_file = data_dir / 'igwind.js'
    turbines_file = data_dir / 'turbines.json'

    # Mapping dictionaries
    streptyp = {
        'E-40': 'E40',
        'E40/5.40': 'E40 5.40',
        'E40 5.4': 'E40 5.40',
        'E66 18.7': 'E66 18.70',
        'E66/18.70': 'E66 18.70',
        'E66.18': 'E66 18.70',
        'E66 20.7': 'E66 20.70',
        'E70/E4': 'E70 E4',
        'E70/20.71': 'E70 E4',
        'E70': 'E70 E4',
        'E-101': 'E101',
        'E 101': 'E101',
        'E115/3.000': 'E115',
        '3.XM': '3XM',
        'V126/3450': 'V126',
    }

    strepher = {
        'ENERCON': 'Enercon',
        'DeWind': 'Dewind',
    }

    # Download and process turbine data
    download_file(igwurl, igw_file)

    with open(igw_file, 'r') as f:
        data = f.read().replace('var officeLayer = ', '')

    with open(turbines_file, 'w') as f:
        f.write(data)

    turbson = json.load(open(turbines_file, 'r'))
    tlst = [place['data'] for place in turbson[1]['places']]
    igw = pd.DataFrame(tlst, columns=['Name', 'Betreiber1', 'Betreiber2', 'n_Anlagen', 'kW', 'Type', 'Jahr', 'x', 'lat',
                                      'lon', 'url', 'Hersteller', 'Nabenhöhe', 'Rotordurchmesser'])

    igw['Type'].replace(streptyp, inplace=True)
    igw['Hersteller'].replace(strepher, inplace=True)

    # Clean Types
    type_kW_mapping = {
        ('E40', 500): 'E40 5.40',
        ('E40', 600): 'E40 6.44',
        ('E66', 1800): 'E66 18.70',
        ('E82', 2300): 'E82 E2',
        ('E115', 3200): 'E115 E2',
        ('M114', 3170): '3.2M114',
    }

    for (type_, kW), new_type in type_kW_mapping.items():
        igw.loc[(igw['Type'] == type_) & (igw['kW'] == kW), 'Type'] = new_type

    # Add details for specific turbine locations
    location_details = {
        'Oberwaltersdorf': {'Type': 'V112', 'Nabenhöhe': '140', 'Rotordurchmesser': '112'},
        'Pretul': {'Type': 'E82 E4', 'Betreiber1': 'Österreichische Bundesforste'}
    }

    for location, details in location_details.items():
        igw.loc[igw['Name'].str.contains(location), list(details.keys())] = list(details.values())

    # Convert columns to appropriate data types
    igw[['Nabenhöhe', 'Rotordurchmesser']] = igw[['Nabenhöhe', 'Rotordurchmesser']].apply(pd.to_numeric,
                                                                                          errors='coerce')

    # Save processed turbine data to CSV
    igw.to_csv(data_dir / 'igwturbines.csv', sep=';', decimal=',', encoding='utf-8', index=False)
    print('Download of wind turbine data complete')

    # Merge wind turbine location data
    bev = gpd.read_file('ROOTDIR/data/buildings/BAU_2200_BETRIEB_P_0.shp')
    bev = bev.loc[bev['F_CODE'] == 2202].to_crs('epsg:4326')
    bev['geometry'] = bev['geometry'].buffer(20)

    igw = gpd.GeoDataFrame(pd.read_csv(data_dir / 'igwturbines.csv', sep=';', decimal=','),
                           geometry=gpd.points_from_xy(igw['lon'], igw['lat']), crs='epsg:4326')

    turbine_locations = gpd.sjoin(bev, igw, how='left', op='contains')
    turbine_locations['ERSTELLDAT'] = pd.to_datetime(turbine_locations['ERSTELLDAT'])
    turbine_locations['Jahr'].fillna(turbine_locations['ERSTELLDAT'].dt.year, inplace=True)
    turbine_locations['Jahr'] = turbine_locations['Jahr'].astype(int)
    turbine_locations['ERSTELLDAT'] = turbine_locations['ERSTELLDAT'].astype(str)
    turbine_locations = turbine_locations[
        ['NAME', 'Name', 'BETREIBER', 'Betreiber1', 'Betreiber2', 'HOEHE', 'Nabenhöhe',
         'Rotordurchmesser', 'kW', 'Hersteller', 'Type', 'Jahr', 'ERSTELLDAT',
         'n_Anlagen', 'url', 'geometry']]
    turbine_locations.to_file(data_dir / 'turbines.shp')

    print('Turbine locations preprocessed')
