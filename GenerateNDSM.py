"""
    Module:: OutherUtils
    Author:: Aravind Harikumar <aravindhari1@gmail.com>
"""

import os
import numpy as np
import gdal
from tifffile import imsave
from ImageProcessor import RasterOperators
from skimage import io, data, img_as_float32, img_as_uint, img_as_ubyte
import fiona
import rasterio
from rasterio.mask import mask
from fiona.crs import from_epsg
from sklearn.preprocessing import Normalizer

def GenerateNDSM():

    ParamsDict = {
        'base_path'         : './BaseFolder/',
        'dem_file_name'     : 'DEM.tif',
        'dsm_file_name'     : 'DSM.tif',
        'ndsm_file_name'    : 'nDSM.tif',
        'ndsm_min_ht'       : 0, # lower limit
        'ndsm_max_ht'       : 20, # upper limit
        'StudyAreaShp'      : '/Mask.shp',
        'HistogramEqualize' : True,
        'crs'               : 'EPSG:32619', # Reference System EPSG  Proj_iD = 'EPSG:32619'
    }
    
    ndsmgen = NDSM_Generator_Class(ParamsDict).GetInstance()
    ndsmgen.ReprojectImage()
    # ndsmgen.CropImageByShpMask()
    ndsmgen.GenerateNDSM()                    

class NDSM_Generator_Class():

    def __init__(self, Params):
        self.base_path          =  Params['base_path']
        self.dem                =  Params['dem_file_name']
        self.dsm                =  Params['dsm_file_name']
        self.ndsm               =  Params['ndsm_file_name']
        self.ndsm_min_ht        =  Params['ndsm_min_ht']
        self.ndsm_max_ht        =  Params['ndsm_max_ht']
        self.crs                =  Params['crs']
        self.StudyAreaShp       =  Params['StudyAreaShp']
        self.HistogramEqualize  =  Params['HistogramEqualize']

    def ReprojectImage(self):
        # Reproject dem
        in_image = os.path.join(self.base_path, self.dem)
        out_image = os.path.join(self.base_path, self.dem)
        RasterOperators.ReprojectImage(in_image, out_image, self.crs)
    
        # Reproject dsm
        in_image = os.path.join(self.base_path, self.dsm)
        out_image = os.path.join(self.base_path, self.dsm)
        RasterOperators.ReprojectImage(in_image, out_image, self.crs)

    def CropImageByShpMask(self):
        with fiona.open(StudyAreaShp, 'r') as shapefile:
            ShapeMask = [feature["geometry"] for feature in shapefile]
            # Crop dem
            RasterOperators.CropImage(os.path.join(self.base_path,self.dem), ShapeMask, os.path.join(self.base_path,self.dem))
            # Crop dsm
            RasterOperators.CropImage(os.path.join(self.base_path,self.dsm), ShapeMask, os.path.join(self.base_path,self.dsm))

    def BandNormalize(self,array):
        array_min, array_max = np.amin(array), np.amax(array)
        return img_as_float32((array - array_min) / (array_max - array_min))

    def GenerateNDSM(self):
        with rasterio.open(os.path.join(self.base_path,self.dem)) as dem_src:
                with rasterio.open(os.path.join(self.base_path,self.dsm)) as dsm_src:
                    meta = dem_src.meta
                    # meta['dtype'] = 'float32'
                    with rasterio.open(os.path.join(self.base_path,self.ndsm), 'w', **meta) as ndsm_dst:
                        for bandId in range(1,dem_src.count+1):

                            dem_band_item = dem_src.read(bandId)
                            dsm_band_item = dsm_src.read(bandId)

                            min_width = np.min([np.shape(dsm_band_item)[0],np.shape(dem_band_item)[0]])
                            min_length = np.min([np.shape(dsm_band_item)[1],np.shape(dem_band_item)[1]])

                            dem_band_item = dem_band_item[0:min_width,0:min_length]
                            dsm_band_item = dsm_band_item[0:min_width,0:min_length]

                            # Generate NDSM
                            ndsm_band_item = np.subtract(dsm_band_item, dem_band_item)
                            # maxheigt = np.amax(ndsm_band_item)

                            # Remove NAN and Outlier Values
                            ndsm_band_item[np.isnan(ndsm_band_item)] = 0
                            ndsm_band_item[ndsm_band_item<self.ndsm_min_ht] = 0
                            ndsm_band_item[ndsm_band_item>self.ndsm_max_ht] = 0
                            
                            # Perform Histogram Equalization
                            if(self.HistogramEqualize):
                                ndsm_band_item = img_as_float32(RasterOperators.HistogramEqualizeImage(ndsm_band_item))
                            
                            # Normalize and Scale
                            ndsm_band_item = self.BandNormalize(ndsm_band_item)*np.finfo(np.float32).max  # or maxheigt

                            ndsm_dst.write_band(bandId, ndsm_band_item)


    def GetInstance(self):
        return self

GenerateNDSM()
