"""
    Platform: Unix, Windows
    Synopsis: A module for raster operations
    Author: Aravind Harikumar <aravindhari1@gmail.com>
"""

import rasterio
import os
import math
import numpy as np
import skimage
from skimage import io, data, img_as_float, img_as_uint
from scipy import ndimage as ndi
from skimage import exposure
import osr
import subprocess
import sys
import rasterstats
from shapely.geometry import Point, LineString, Polygon, mapping
import geopandas as gpd
import osr
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona
from fiona.crs import from_epsg
from osgeo import gdal, gdalconst, ogr, osr

def GetImageInfo(InDataPath):
    dataset = rasterio.open(InDataPath)
    print('Reading ' + dataset.name)
    print(' - Width:' + str(dataset.width) + '\n' +
          ' - Height:' + str(dataset.height) + '\n' + 
          ' - Bands:' + str(dataset.count) + '\n' + 
        #   ' - Datatype:' + str(type(dataset.uint8)) + '\n' +
          ' - Projection:' + str(dataset.crs) + '\n' + 
          ' - Transform:' + str(dataset.transform) + '\n' +
          ' - bounds:'+ str(dataset.bounds))
          
    return dataset

def ReadImage(InDataPath):
    dataset = rasterio.open(InDataPath)
    print('Reading ' + dataset.name)
    print('Width:' + str(dataset.width) + ' Height:' + str(dataset.height) + ' Bands:' + str(dataset.count))
    return dataset

def CopyImage(InFileName,OutFileName):    
    # Read metadata of first file
    with rasterio.open(InFileName) as src0:
        meta = src0.meta
        # Read each layer and write it to stack
        with rasterio.open(OutFileName, 'w', **meta) as dst:
            for i in range(1,src0.count+1):
                dst.write_band(i, src0.read(i))
    
    print('Done Copying!')
    return True

def CopySmoothenImage(InFileName,OutFileName, smoothig_factor):    
    # Read metadata of first file
    with rasterio.open(InFileName) as src0:
        meta = src0.meta
        meta.update(dtype = 'uint16')
        # meta.update(nodata = 1)

        # Read each layer and write it to stack
        with rasterio.open(OutFileName, 'w', **meta) as dst:
            for i in range(1,src0.count+1):
                band_arr = src0.read(i)
                # print(np.amin(band_arr))
                # exit(0)
                band_arr = band_arr #+ np.amin(band_arr)
                band_arr = skimage.filters.gaussian(band_arr, sigma=smoothig_factor)
                band_arr *= 255.0/band_arr.max() 
                dst.write_band(i, band_arr.astype('uint16'))
    
    print('Done Copying!')
    return True

def StackImageBands(file_list,OutFileName):    
    # Read metadata of first file
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count = len(file_list))

    # Read each layer and write it to stack
    with rasterio.open(OutFileName, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))
    
    print('Done stacking!')
    return True


def StackSelectedImageBands(bandlist, in_file_name, OutFileName, rs_data_type):    
    # Read metadata of first file
    with rasterio.open(in_file_name) as src0:
        meta = src0.meta.copy()
        # Update meta to reflect the number of layers
        meta.update(count = len(bandlist))
        meta.update(wavelength = bandlist)
        # print(meta)
        # kwds = src0.profile
        # print(kwds)
    bandcnt = 1
    # Read each layer and write it to stack
    with rasterio.open(OutFileName, 'w', **meta) as dst:
        with rasterio.open(in_file_name) as src1:

            if (rs_data_type.lower() == 'mulispectral'): # for multispectral
                bandlist = common_elements(meta['wavelength'], str(bandlist))
            # else:
                # bandlist = bandlist

            for i in range(1,src1.count):                    
                # # band[math.isnan(band)] = 0
                if (rs_data_type.lower() == 'mulispectral'):
                    metadata = (meta['wavelength'])[0]  # for multispectral  
                else:
                    metadata = src1.tags(i)['wavelength']  # for hyperspectral  

                if(metadata in str(bandlist)):
                    band = src1.read(i)  
                    # print(metadata)
                    dst.write_band(bandcnt, src1.read(i))
                    bandcnt = bandcnt + 1

def common_elements(list1, list2):
    return [element for element in list1 if element in list2]

def pixel2coord(raster, x, y):
    xoff, a, b, yoff, d, e = raster.GetGeoTransform()
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return(xp, yp)

def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
            # print(x,y)
        yarr.reverse()
    return ext

def NormalizeList(list_data):
     return [float(i)/max(list_data) for i in list_data]

def GetLargestN(arr, n):
    """Returns the n largest indices from a numpy array."""
    flat = arr.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, arr.shape), arr[np.unravel_index(indices, arr.shape)]

def newproj(inputImage,outputImage,gcp_list, out_file_datatype):

    dataset = gdal.Open(inputImage) 
    I = dataset.ReadAsArray(0,0,dataset.RasterXSize,dataset.RasterYSize)
    # print(dtype(I))
    # exit(0)
    # x = I[~numpy.isnan(I)]
    outdataset = gdal.GetDriverByName('GTiff') 
    output_SRS = osr.SpatialReference() 
    output_SRS.ImportFromEPSG(32619) 

    if(len(I.shape)>2):
        outdataset = outdataset.Create(outputImage,dataset.RasterXSize,dataset.RasterYSize,I.shape[0],out_file_datatype) 
        for nb_band in range(0,I.shape[0]):
            # print(nb_band)
            outdataset.GetRasterBand(nb_band+1).WriteArray(I[nb_band,:,:])
    else:
        outdataset = outdataset.Create(outputImage,dataset.RasterXSize,dataset.RasterYSize,1,gdal.GDT_Float32)
        for nb_band in range(0,1):
            # row_sums = I.sum(axis=1)
            # I = I / row_sums[:, np.newaxis]
            # I = I * 65535
            outdataset.GetRasterBand(nb_band+1).WriteArray(I)

    outdataset.SetProjection(output_SRS.ExportToWkt()) 
    wkt = outdataset.GetProjection() 
    outdataset.SetGCPs(gcp_list,wkt)
    # print('Wrap')
    gdal.Warp(outputImage, outdataset, dstSRS='EPSG:32619')

    outdataset = None

def ReprojectImageCoreg(input_image, outputimage, points, dst_crs):
    # dst_crs = 'EPSG:32619'

    with rasterio.open(input_image) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        print('widdth' + str(width), str(height))
        kwargs = src.meta.copy()
        # print(kwargs)
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(outputimage, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    # src_transform=src.transform,
                    gcps=points,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def ReprojectImage(input_image, outputimage, dst_crs):
    # dst_crs = 'EPSG:32619'

    with rasterio.open(input_image) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        print(kwargs)
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(outputimage, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=None,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest)


def CropImage(InputImage, ShpFile, OutFileName):
    with rasterio.open(InputImage) as ImageObj:
        out_image, out_transform = rasterio.mask.mask(ImageObj, ShpFile, crop=True, filled=True, nodata = 0)
        out_meta = ImageObj.meta.copy()
        out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

    # with rasterio.open(OutFileName+".tif", "w", **out_meta) as dest:
    with rasterio.open(OutFileName, "w", **out_meta) as dest:
        dest.write(out_image)

def CropImageForShpObj(InputImage, CrownBuffer, OutFileName):
    with rasterio.open(InputImage) as ImageObj:
        out_image, out_transform = rasterio.mask.mask(ImageObj, gpd.GeoSeries(Polygon(CrownBuffer)), crop=True, filled=True, nodata = 0)
        out_meta = ImageObj.meta.copy()
        out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

    # with rasterio.open(OutFileName+".tif", "w", **out_meta) as dest:
    with rasterio.open(OutFileName, "w", **out_meta) as dest:
        dest.write(out_image)

def CropImageForShpObjAndEqualize(InputImage, CrownBuffer, OutFileName):
    with rasterio.open(InputImage) as ImageObj:
        out_image, out_transform = rasterio.mask.mask(ImageObj, gpd.GeoSeries(Polygon(CrownBuffer)), crop=True, filled=True, nodata = 0)
        out_meta = ImageObj.meta.copy()
        out_meta.update({"driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "dtype": 'uint16'}) #uint16, float64
        # out_meta.update(dtype = 'float64')
    out_image = HistogramEqualizeImage(skimage.img_as_float(out_image))
    out_image = skimage.img_as_uint(out_image)
    # with rasterio.open(OutFileName+".tif", "w", **out_meta) as dest:
    with rasterio.open(OutFileName, "w", **out_meta) as dest:
        dest.write(out_image)

def GetCropImage(InputImage, ShpFile):
    with rasterio.open(InputImage) as ImageObj:
        out_image, out_transform = rasterio.mask.mask(ImageObj, ShpFile, crop=True, filled=True, nodata = 0)
        return out_image

def HistogramEqualizeImage(InputImage):
	return img_as_float(exposure.equalize_hist(InputImage))


def MaxFilterImage(InputImage,FilterSize):
    return ndi.maximum_filter(InputImage, size=FilterSize, mode='constant')


def GaussianFilterImage(InputImage,sigmavalue,truncvalue):
    return skimage.filters.gaussian(InputImage, sigma=sigmavalue, truncate=truncvalue)

def array2raster(self,rasterfn,newRasterfn,array):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def ReadImageSKT(InDataPath):    
    # ndom_image = np.random.random([500,500])
    img = io.imread(InDataPath) # , key=0
    return img

def WriteImageSKT(Datapacket, OutDataPath):
    io.imsave(OutDataPath, Datapacket)

def ShowImageSKT(Datapacket):
    io.imshow(Datapacket)
    io.show()

# Normalize bands into 0.0 - 1.0 scale
def normalize(array):
    # array[array==-9999]=0
    array_min, array_max = array.min(), array.max()
    return img_as_float((array - array_min) / (array_max - array_min))

def __gdalwarp(*args):
    return subprocess.check_call(['gdalwarp'] + list(args))

def TileImage(src_path, out_base, id1, SpanImage, StudyAreaShp):

    ds = gdal.Open(SpanImage)

    gt = ds.GetGeoTransform()

    width_px = ds.RasterXSize
    height_px = ds.RasterYSize

    # Get coords for lower left corner 17796 21564
    xmin = int(gt[0])
    xmax = int(gt[0] + (gt[1] * width_px))

    # get coords for upper right corner
    if gt[5] > 0:
        ymin = int(gt[3] - (gt[5] * height_px))
    else:
        ymin = int(gt[3] + (gt[5] * height_px))

    ymax = int(gt[3])
    # print(xmax,xmin)
    # split height and width into four - i.e. this will produce 25 tiles
    tile_width = (xmax - xmin) // 6
    tile_height = (ymax - ymin) // 5
    # print(tile_width)
    for x in range(xmin, xmax, tile_width):
        for y in range(ymin, ymax, tile_height):
            __gdalwarp('-te', str(x), str(y), str(x + tile_width),
                    str(y + tile_height), '-multi', '-overwrite', '-wo', 'NUM_THREADS=ALL_CPUS',
                    '-wm', '2500', src_path, os.path.join(out_base, '{}_{}_{}.tif'.format(id1, x, y)))
            
            OrthoImageName = os.path.join(out_base, '{}_{}_{}.tif'.format(id1, x, y))
            # print(OrthoImageName)
            # OrthoImageNameO = out_base  + 'temp/' + '{}_{}_{}.tif'.format('s', x, y)
            with fiona.open(StudyAreaShp, 'r') as shapefile:
                ShapeMask = [feature["geometry"] for feature in shapefile]
                try:
                    CropImage(OrthoImageName, ShapeMask, OrthoImageName)
                    ds0 = gdal.Open(OrthoImageName)
                    band = ds0.GetRasterBand(1)
                    oop = np.mean(band.ReadAsArray())
                    # print(oop)
                    if(oop==0):
                        if os.path.exists(OrthoImageName):
                            os.remove(OrthoImageName)
                            print('Removed non overlapping images00')
                except Exception as w:
                    print(str(w))
                    if os.path.exists(OrthoImageName):
                        os.remove(OrthoImageName)
                        print('Removed non overlapping images')


def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
            # print(x,y)
        yarr.reverse()
    return ext

def AlignRasterNDSM(inputfile, referencefile, outputfile):

    input_file = gdal.Open(inputfile,gdalconst.GA_ReadOnly)
    inputProj = input_file.GetProjection()
    # inputTrans = input_file.GetGeoTransform()

    reference = gdal.Open(referencefile, gdalconst.GA_ReadOnly)
    referenceProj = reference.GetProjection()
    referenceTrans = reference.GetGeoTransform()
    # bandreference = reference.GetRasterBand(1) # 6643 6605

    # dst_filename = out_ndsm_org_res_file 
    # dst_or = gdal.GetDriverByName('GTiff').Create(dst_filename, 6643, 6605, 1, gdalconst.GDT_Float32)
    # dst_or.SetGeoTransform( referenceTrans )
    # dst_or.SetProjection( referenceProj)
    # gdal.ReprojectImage(input_file, dst_or, inputProj, referenceProj, gdalconst.GRA_NearestNeighbour)

    dst_filename = outputfile
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, reference.RasterXSize, reference.RasterYSize, 1, gdalconst.GDT_UInt16)
    dst.SetGeoTransform( referenceTrans )
    dst.SetProjection( referenceProj)
    gdal.ReprojectImage(input_file, dst, inputProj, referenceProj, gdalconst.GRA_NearestNeighbour)

def AlignRaster(inputfile, referencefile, outputfile, out_ndsm_org_res_file, outputfileOrtho):

    input_file = gdal.Open(inputfile,gdalconst.GA_ReadOnly)
    inputProj = input_file.GetProjection()
    # inputTrans = input_file.GetGeoTransform()

    reference = gdal.Open(referencefile, gdalconst.GA_ReadOnly)
    referenceProj = reference.GetProjection()
    referenceTrans = reference.GetGeoTransform()
    # bandreference = reference.GetRasterBand(1) # 6643 6605

    # dst_filename = out_ndsm_org_res_file 
    # dst_or = gdal.GetDriverByName('GTiff').Create(dst_filename, 6643, 6605, 1, gdalconst.GDT_Float32)
    # dst_or.SetGeoTransform( referenceTrans )
    # dst_or.SetProjection( referenceProj)
    # gdal.ReprojectImage(input_file, dst_or, inputProj, referenceProj, gdalconst.GRA_NearestNeighbour)

    dst_filename = outputfile
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, reference.RasterXSize, reference.RasterYSize, 1, gdalconst.GDT_UInt16)
    dst.SetGeoTransform( referenceTrans )
    dst.SetProjection( referenceProj)
    gdal.ReprojectImage(input_file, dst, inputProj, referenceProj, gdalconst.GRA_NearestNeighbour)

    dst_filename_ortho = outputfileOrtho
    dst_ortho = gdal.GetDriverByName('GTiff').Create(dst_filename_ortho, reference.RasterXSize, reference.RasterYSize, reference.RasterCount, gdalconst.GDT_UInt16)
    dst_ortho.SetGeoTransform( referenceTrans )
    dst_ortho.SetProjection( referenceProj) 
    gdal.ReprojectImage(reference, dst_ortho, inputProj, referenceProj, gdalconst.GRA_NearestNeighbour)
    # del dst

    # ext = GetExtent(referenceTrans,reference.RasterXSize, reference.RasterYSize)
    # print(ext)
    # Xspan = ext[2][0] - ext[0][0]
    # Yspan = ext[0][1] - ext[2][1]
    # x_res = Xspan/reference.RasterXSize
    # y_res = Xspan/reference.RasterYSize
    # print(x_res,y_res)
    # OutTile = gdal.Warp(dst, input_file, format='GTiff', outputBounds=[ext[2][0], ext[0][1], ext[0][0], ext[2][1]], 
    # xRes=x_res, yRes=y_res, dstSRS=referenceProj, resampleAlg=gdal.GRA_NearestNeighbour)

def GetNDVIMap(OrthoPhoto, OutputNDVIFile, NIRBandIndex, RBandIndex):
    with rasterio.open(os.path.join(OrthoPhoto)) as ortho_src:
        meta = ortho_src.meta
        meta.update(count = 1)
        meta.update(dtype = 'float64')
        meta.update(width = ortho_src.width)
        meta.update(height = ortho_src.height)

        NIR = ortho_src.read(NIRBandIndex).astype(float)
        R = ortho_src.read(RBandIndex).astype(float)

        with rasterio.open(os.path.join(OutputNDVIFile), 'w', **meta) as ndvi_dst:
            numerator = np.subtract(NIR, R)
            denominator = np.add(NIR, R)
            NDVIMat = np.divide(numerator,denominator)
            # print(np.sum(np.isnan(NDVIMat)))
            NDVIMat[np.isnan(NDVIMat)] = 0            
            # exit(0)
            ndvi_dst.write_band(1, NDVIMat)

def GetPRIMap(OrthoPhoto, OutputPRIFile, band528Index, band570Index):
    with rasterio.open(os.path.join(OrthoPhoto)) as ortho_src:
        meta = ortho_src.meta
        meta.update(count = 1)
        meta.update(dtype = 'float64')
        meta.update(width = ortho_src.width)
        meta.update(height = ortho_src.height)

        b528 = ortho_src.read(band528Index).astype(float)
        b570 = ortho_src.read(band570Index).astype(float)
        with rasterio.open(os.path.join(OutputPRIFile), 'w', **meta) as pri_dst:

            numerator = np.subtract(b528, b570)
            denominator = np.add(b528, b570)
            PRIMat =  np.divide(numerator,denominator)
            PRIMat[np.isnan(PRIMat)] = 0
            pri_dst.write_band(1, PRIMat)

def GetCCIMap(OrthoPhoto, OutputCCIFile, band528Index, band645Index):
    with rasterio.open(os.path.join(OrthoPhoto)) as ortho_src:
        meta = ortho_src.meta
        meta.update(count = 1)
        meta.update(dtype = 'float64')
        meta.update(width = ortho_src.width)
        meta.update(height = ortho_src.height)
        
        b528 = ortho_src.read(band528Index).astype(float)
        b645 = ortho_src.read(band645Index).astype(float)

        with rasterio.open(os.path.join(OutputCCIFile), 'w', **meta) as cci_dst:
            numerator = np.subtract(b528, b645)
            denominator = np.add(b528, b645) 
            CCIMat =  np.divide(numerator,denominator)
            CCIMat[np.isnan(CCIMat)] = 0
            cci_dst.write_band(1, CCIMat)
