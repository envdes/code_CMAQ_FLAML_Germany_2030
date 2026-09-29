"""
CMAQ_raw_nc_to_csv.py
=====================

Re-project and clip raw CMAQ concentration output (Lambert Conformal Conic grid)
onto a regular lon/lat grid and store the result as a NetCDF file.

Every hourly layer is:
    1. written to a temporary GeoTIFF with the LCC definition of the CMAQ grid,
    2. warped to EPSG:4326 with bilinear resampling,
    3. clipped with a region shapefile (Germany),
    4. stacked into a (time, lat, lon) cube.
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import os

import numpy as np
import pandas as pd
import xarray as xr
from osgeo import osr, gdal


# ----------------------------------------------------------------------------
# Re-project and clip one species of a raw CMAQ file
# ----------------------------------------------------------------------------
def cmaqClipByGdal(year, infile, outfile, spec):
    """
    Resample and re-project the input file with gdal.Warp and write the result
    to `outfile` as a NetCDF file.
    """
    # Hours of spin-up padding repeated at the beginning / end of the time axis.
    stntime = 24
    endntime = 24 * 2

    input_nc = xr.open_dataset(infile, engine='netcdf4')

    print("deal with: %s" % spec)
    dataset = input_nc[spec]
    print(dataset.shape)
    att = input_nc.attrs
    Bands = dataset.shape[0] + stntime + endntime

    # Geotransform definition of the raw CMAQ grid.
    N_Lon = dataset.shape[3]
    N_Lat = dataset.shape[2]
    Lon_Res = att['XCELL']
    Lat_Res = att['YCELL']
    LonMin = att['XORIG']
    LonMax = LonMin + Lon_Res * (N_Lon - 1)
    LatMin = att['YORIG']
    LatMax = LatMin + Lat_Res * (N_Lat - 1)
    geotransform = (LonMin, Lon_Res, 0, LatMax, 0, -Lat_Res)
    print(geotransform)

    # Projection definition: Lambert Conformal Conic on a sphere.
    lcc = osr.SpatialReference()
    lcc.ImportFromProj4('+proj=longlat +a=6370000 +b=6370000 +no_defs')
    lcc.SetLCC(40.0, 53.0, 46.5, 12.0, 0, 0)

    # Loop over all output bands; the first / last 24 hours repeat the first /
    # last hours of the source file so that the padded time axis can be filled.
    for j in range(Bands):

        if j <= stntime:
            k = j
        elif j >= (Bands - endntime):
            k = j - stntime - endntime
        else:
            k = j - stntime

        # Raster rows run north -> south, so the CMAQ array is flipped.
        arr0 = np.flipud(dataset[k, 0, :, :])

        # Write the temporary GeoTIFF of the current hour.
        driver = gdal.GetDriverByName('GTiff')
        tmp_tif = driver.Create(outfile + '_tmp1', N_Lon, N_Lat, 1, gdal.GDT_Float32)
        tmp_tif.SetGeoTransform(geotransform)
        tmp_tif.SetProjection(lcc.ExportToWkt())
        tmp_tif.GetRasterBand(1).WriteArray(arr0)

        tif_clip(tmp_tif, outfile + '_tmp2')

        tem_ds = gdal.Open(outfile + '_tmp2')
        tem_arr = tem_ds.ReadAsArray()

        # The clipped geometry is the same for every band, so the output
        # container is only built once.
        if j == 0:
            nXSize = tem_ds.RasterXSize
            nYSize = tem_ds.RasterYSize
            print(nXSize, nYSize)
            im_geotrans = tem_ds.GetGeoTransform()
            print(im_geotrans)
            lonmin = im_geotrans[0]
            lonmax = im_geotrans[0] + im_geotrans[1] * (nXSize - 1)
            latmax = im_geotrans[3]
            latmin = im_geotrans[3] + im_geotrans[5] * (nYSize - 1)

            date_range = pd.date_range('%d-01-01-05' % year, periods=Bands, freq='h')

            ds_out = xr.Dataset.from_dict({
                "coords": {
                    "lat": {"dims": ("lat",),
                            "attrs": {"standard_name": "latitude", "units": "0.1 degrees", "axis": "Y"},
                            "data": np.linspace(latmax, latmin, nYSize), },
                    "lon": {"dims": ("lon",),
                            "attrs": {"standard_name": "longitude", "units": "0.1 degrees", "axis": "X"},
                            "data": np.linspace(lonmin, lonmax, nXSize), },
                    "time": {"dims": ("time",),
                             "attrs": {"standard_name": "time", "long_name": "Time daily"},
                             "data": date_range}},
                "data_vars": {spec: {"dims": ("time", "lat", "lon"),
                                     "attrs": input_nc[spec].attrs,
                                     "data": np.zeros((Bands, nYSize, nXSize), dtype='float32')}}
            })

        ds_out[spec][j, :, :] = np.array(tem_arr)
        tmp_tif = None

    os.remove(outfile + '_tmp1')
    os.remove(outfile + '_tmp2')

    ds_out.to_netcdf(outfile, mode='w', format="NETCDF3_CLASSIC", engine='netcdf4')


# ----------------------------------------------------------------------------
# Clip and re-sample a raster with gdal.Warp
# ----------------------------------------------------------------------------
def tif_clip(stiff, outfile):
    """
    :param stiff: source raster
    :param outfile: output tif file path
    :return: clip and re-sampled tif file
    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    # First pass: re-project to WGS84 with bilinear resampling.
    gdal.Warp(outfile,
              stiff,
              format='GTiff',
              dstSRS=srs,
              resampleAlg=gdal.GRA_Bilinear,
              multithread=True,
              outputType=gdal.GDT_Float32
              )

    # Second pass: clip to the region of interest.
    stiff1 = gdal.Open(outfile)
    gdal.Warp(outfile,
              stiff1,
              format='GTiff',
              dstSRS=srs,
              resampleAlg=gdal.GRA_Bilinear,
              multithread=True,
              cutlineDSName="/home/qinjialing/map/germany.shp",
              cropToCutline=True,
              outputType=gdal.GDT_Float32
              )
    return outfile


# ----------------------------------------------------------------------------
# Run the conversion
# ----------------------------------------------------------------------------
indirs = "/home/qinjialing/output_ssp245"
outdirs = "/home/qinjialing/output_ssp245"
infile = os.path.join(indirs, '36km.CMAQ.2030.conc')
year = 2030

for spec in ['O3']:
    outfile = os.path.join(outdirs, "test_%s_245.nc" % (spec))
    print('deal with %s' % spec)

    cmaqClipByGdal(year, infile, outfile, spec)
