


def world2Pixel(geoMatrix, (x,y)):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate.

    ----------
    Parameters
    ----------

    - `geoMatrix`: (`tuple`, `list` or `array`)
        the GeoTransformMatrix extracted from the reference raster which is:
        (lonMin, lonStep, 0, latMax, 0, latStep)

    - `(x,y)`: (`tuple`, `list` or `array`)
        the (lon, lat) coordinates of the point to transform.

    -------
    Returns
    -------

    (pixel, line) (tuple)

        The (column, row)=(j, i) index of raster cell containing the geographical point.
        Here we use the upper left notation so that the first index corresponds to the longitude
        from west to east (x-axis while it is the column index actually) and the second one corresponds
        to the latitude from north to south (y-axis while it is actually the row index of the array).
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    pixel = int(round((x - ulX) / xDist))
    line = int(round((y - ulY) / yDist))
    return (pixel, line)

def pixel2World(geoMatrix, (x,y)):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the geospatial coordinate of a pixel location.

    ----------
    Parameters
    ----------

    - `geoMatrix`: (`tuple`, `list` or `array`)
        the GeoTransformMatrix extracted from the reference raster which is:
        (lonMin, lonStep, 0, latMax, 0, latStep)

    - `(x,y)`: (`tuple`, `list` or `array`)
        the (column, row) = (j, i) coordinates of the pixel to transform (here we work in the upper left
        notation, see output for explanation).

    -------
    Returns
    -------

    (lon, lat) (tuple)

        The (longitude, latitude) values of the upper left corner of the raster cell at poisition (x,y),
        i.e., at row y and column x of the original array.
        Here we use the upper left notation so that the first index corresponds to the longitude
        from west to east (x-axis while it is the column index actually) and the second one corresponds
        to the latitude from north to south (y-axis while it is actually the row index of the array).
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]

    lat = ulY + y*yDist
    lon = ulX + x*xDist

    return (lon, lat)



import gdal
import gdalnumeric as gdaln
import numpy as np
from  numpy import dtype
def array2raster(newRasterfn, originalRasterfn, usrArray, mergeFunction=lambda o,a: a, offset=(0,0),\
                 originalRasterBand=1, noDataValue=-999, outDataType=None, postProcessFilter=None):
    '''
    Creates a new raster in the `newRasterfn` file by using the same geoTransform matrix and projection of
    the original raster in `originalRasterfn`. Creates a new raster which is a subset of the original one 
    and has the shape of `array`. If `array` is smaller than raster and it is referring to a inner part of
    the original raster you can specify the offset as the upper left corner in `offset=(i_offset, j_offset)`.
    
    The new raster is computed as a merge function `mergeFunction(original, array)` (default is that it copies)
    the `array` into the new raster. If you want to use the original raster as a weight for your array you can do
    `mergeFunction = lambda(o,a): o*a`.
    
    If `postProcessFilter` is specified we further enforce the no data value to the data complying with it. It is
    a function taking as arguments the reference raster value, the original array value and the newly computed value,
    and returning `True` where we want the noDataValue for fine tuning the output filter.
    For instance, to set all the negative cells of the newly computed array to `noDataValue` one can do

    ```
    postProcessFilter = lambda reference, original, new: new<0
    ```

    The output type can be set as one of the gdal.GDT_*** types.
    '''
    # Load the original raster to get information...
    originalRaster = gdal.Open(originalRasterfn)
    transfMatrix = originalRaster.GetGeoTransform()
    originalProjection = originalRaster.GetProjectionRef()
    originalBand = originalRaster.GetRasterBand(originalRasterBand)
    originalBandNoValue = originalBand.GetNoDataValue()

    # Translate the information for the upper-left corner
    originX, originY, stepX, stepY = transfMatrix[0], transfMatrix[3], transfMatrix[1], transfMatrix[5]
    ulPxX, ulPxY = offset[0], offset[1]
    ulGeoX, ulGeoY = pixel2World(transfMatrix, offset)

    userArray = np.asarray(usrArray)
    cols, rows = usrArray.shape[1], usrArray.shape[0]

    # Load the original reference raster already cutting it to the desired edges...
    originalArray = np.asarray(gdaln.LoadFile(originalRasterfn, xoff=ulPxX, yoff=ulPxY, xsize=cols, ysize=rows))
    print "Reference stats:", originalArray.min(), originalArray.max(), originalArray.dtype, originalArray.shape
    print "User def. stats:", usrArray.min(), usrArray.max(), usrArray.dtype, usrArray.shape

    # Evaluate the output and delete reference to original raster...
    dumpArray = mergeFunction(originalArray, userArray)
    #dumpArray[originalArray == originalBandNoValue] = noDataValue

    if postProcessFilter:
        dumpArray[postProcessFilter(originalArray, userArray, dumpArray)] = noDataValue
        
    originalArray = None
    del originalArray

    print "Trasnform stats:", dumpArray.min(), dumpArray.max(), dumpArray.dtype, dumpArray.shape

    np2gdalType = {dtype("float32"): gdal.GDT_Float32, dtype("float64"): gdal.GDT_Float64,\
                   dtype("int32"): gdal.GDT_Int32, dtype("int16"): gdal.GDT_Int16}
    if not outDataType:
        outDataType = gdal.GDT_Float64
        print "WARNING, no `outDataType` selected, using default one: GDT_Float32", "->", outDataType,
        if outDataType != np2gdalType[dumpArray.dtype]:
            print "that differs from the dump array one", dumpArray.dtype, "->", np2gdalType[dumpArray.dtype]
    elif outDataType != np2gdalType[dumpArray.dtype]:
        print "WARNING, selected `outDataType`", outDataType, "differs from the dump array one",\
                    dumpArray.dtype, "->", np2gdalType[dumpArray.dtype]

    # Get the driver and create the tiff...
    driver = gdal.GetDriverByName("GTiff")
    target_tif = driver.Create(newRasterfn, cols, rows, bands=1, eType=outDataType)
    band = target_tif.GetRasterBand(1)
    band.WriteArray(dumpArray)
    band.FlushCache()
    band.SetNoDataValue(noDataValue)
    del dumpArray

    # Copying the reference and updating the Trasforming matrix...
    target_tif.SetGeoTransform((ulGeoX, stepX, 0, ulGeoY, 0, stepY))
    target_tif.SetProjection(originalProjection)

    # Closing all the open streams and buffers...
    originalRaster = None
    originalBand = None
    band = None
    target_tif = None

import ogr
import osr
import shapefile
def shape2raster(outRasterFn, shapeFn, shapeField, rasterFn, rasterBand=1, noDataValue=-999,\
                 mergeFunction=None, outDataType=None, rasterSize=None, postProcessFilter=None):

    # Load the shapefile and locate its boundaries
    shp_handle = ogr.Open(shapeFileName)
    shp_layer = shp_handle.GetLayer(0)
    shp_lonmin, shp_lonmax, shp_latmin, shp_latmax = shp_layer.GetExtent()

    # Load the raster and fetch the georeference matrix
    originalRaster = gdal.Open(rasterFn)
    transfMatrix = originalRaster.GetGeoTransform()
    originalProjection = originalRaster.GetProjectionRef()
    originalBand = originalRaster.GetRasterBand(rasterBand)

    # Translate the information for the upper-left corner
    originX, originY, stepX, stepY = transfMatrix[0], transfMatrix[3], transfMatrix[1], transfMatrix[5]

    # Here we fix the actual boundaries of the shapefile...
    shape_pxul, shape_pxlr = world2Pixel(transfMatrix, (shp_lonmin, shp_latmax)),\
                        world2Pixel(transfMatrix, (shp_lonmax, shp_latmin))
    shape_ul = pixel2World(transfMatrix, shape_pxul)
    shape_lr = pixel2World(transfMatrix, shape_pxlr)
    shapeWidth, shapeHeight = shape_lr[0]-shape_ul[0], shape_lr[1]-shape_ul[1]

    if rasterSize:
        x_resolution, y_resolution = rasterSize
    else:
        x_resolution, y_resolution = int(round(shapeWidth/stepX)), int(round(shapeHeight/stepY))

    print (shp_lonmin, shp_latmax), (shp_lonmax, shp_latmin)
    print shape_ul, shape_lr
    print shape_pxul, shape_pxlr
    print "shape delta lon, delta lat", shapeWidth, shapeHeight
    print "shape delta Xpx, Ypx", x_resolution, y_resolution
    print

    # Rasterize the shapefile accordingly on the selected field
    tmpRasterFn = "._mytmp_name_difficult_to_replicate.tif"

    driver = gdal.GetDriverByName("GTiff")
    target_tif = driver.Create(tmpRasterFn, x_resolution, y_resolution, bands=1, eType=gdal.GDT_Float64)
    target_tif.SetGeoTransform([shape_ul[0], stepX, .0, shape_ul[1], .0, stepY])
    target_tif.SetProjection(originalProjection)
    band = target_tif.GetRasterBand(1)
    band.SetNoDataValue(noDataValue)
    err = gdal.RasterizeLayer(target_tif, [1], shp_layer, None, None, [1],\
                              ["ALL_TOUCHED=TRUE", "ATTRIBUTE=%s" % shapeField])
    if err:
        raise RuntimeError, "Error rasterizing attribute >> %s << !" % shapeField
    print shapeField
    target_tif = None

    if mergeFunction is None:
        os.rename(tmpRasterFn, outRasterFn)
    else:
        # Merge the two rasters with the provided function
        shapeArray = gdaln.LoadFile(tmpRasterFn)

        # Write out the raster
        array2raster(outRasterFn, rasterFn, shapeArray, mergeFunction=mergeFunction,\
                     offset=shape_pxul, originalRasterBand=rasterBand, noDataValue=noDataValue,\
                     outDataType=outDataType, postProcessFilter=postProcessFilter)
        os.remove(tmpRasterFn)

def getRasterInfo(rasterFileName, originalRasterBand=1):
    originalRaster = gdal.Open(rasterFileName)
    transfMatrix = originalRaster.GetGeoTransform()
    rasterShape = (originalRaster.RasterYSize, originalRaster.RasterXSize)
    originalProjection = originalRaster.GetProjectionRef()
    originalBand = originalRaster.GetRasterBand(originalRasterBand)
    originalBandNoValue = originalBand.GetNoDataValue()

    return transfMatrix, originalProjection, rasterShape, originalBandNoValue

def shape2box(shape, shapeType=None):
    '''
    Returns `[[lonMin, lonMax], [latMin, latMax]]`.
    '''

    tmp_type = shapeType if shapeType else shape["type"]
    box = [[180., -180.], [90., -90.]]
    if tmp_type == "MultiPolygon":
        for polygon in shape["coordinates"]:
            tmp_box = shape2box({"coordinates": polygon}, shapeType="Polygon")
            box[0] = [min(tmp_box[0][0], box[0][0]), max(tmp_box[0][1], box[0][1])]
            box[1] = [min(tmp_box[1][0], box[1][0]), max(tmp_box[1][1], box[1][1])]
    elif tmp_type == "Polygon":
        points = np.array([p for p in shape["coordinates"][0]])
        box = [[points[:,0].min(), points[:,0].max()], [points[:,1].min(), points[:,1].max()]]
    else:
        raise RuntimeError
    return box

