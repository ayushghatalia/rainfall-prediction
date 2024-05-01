import ee
import geemap
import numpy as np
import glob
# For working with era5 nc files and shapefiles
import xarray as xr
import rasterio as rst
from rasterio.plot import show

def extract_data():
    ee.Authenticate()
    ee.Initialize()

    aoi=ee.Geometry.Polygon(
            [[[85.15180711804797, 29.014851679824427],
                [85.15180711804797, 22.947200895054186],
                [97.86811571179796, 22.947200895054186],
                [97.86811571179796, 29.014851679824427]]])
    era5l = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filter(ee.Filter.date('2011-01-01', '2020-12-31')).filter(ee.Filter.calendarRange(6, 9,'month')).filterBounds(aoi).select(['dewpoint_temperature_2m','skin_temperature','surface_net_solar_radiation_sum','u_component_of_wind_10m','v_component_of_wind_10m','surface_pressure','total_precipitation_sum'])
    print(era5l.size().getInfo())

    Map = geemap.Map()
    Map.addLayer(aoi,{'color':'FF0000'},'AOI')
    Map.setCenter(90,25,6)
    Map.addLayer(era5l.first().clip(aoi).select(['total_precipitation_sum']), {'min': 0,'max': 0.1,'palette':['Teal','DarkGreen','Chartreuse','yellow','grey','white']}, 'Classified LULC')
    Map.addLayerControl()

    #Exporting the dataset
    geemap.ee_export_image_collection(era5l, '/content/drive/MyDrive/PS1 - NESAC/Dataset', scale=11132, crs=None, region=aoi, file_per_band=False)

def get_data():
    files = glob.glob("/content/drive/MyDrive/PS1/Dataset/*.tif")
    for i in range(len(files)):
        tmp = rst.open(files[i])
        tmp_np = tmp.read()
        tmp_resh = tmp_np.reshape((1,tmp_np.shape[0],tmp_np.shape[1],tmp_np.shape[2]))
        if i == 0:
            data = np.copy(tmp_resh)
        else:
            data = np.append(data,tmp_resh,axis=0)
    indices = np.where(data[0][-1] == 9999)
    print(indices)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j][indices] = 0
    return data

# np.save("/content/drive/My Drive/ERA5 TIFF DATA/NUMPY FILES/2016-2020/im_np.npy", im_np)

