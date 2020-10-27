"""
    Platform: Unix, Windows
    Synopsis: A module for geneating Normalized Digital Surface Model.
    Author: Aravind Harikumar <aravindhari1@gmail.com>
"""

import os
import numpy as np
import gdal
from tifffile import imsave
import RasterOperators
from skimage import io, data, img_as_float32, img_as_uint, img_as_ubyte
import fiona
import rasterio
from rasterio.mask import mask
from fiona.crs import from_epsg
from sklearn.preprocessing import Normalizer

def GenerateNDSM():
    """
    Gereate Normalized Digital Surface Model (NDSM).
    """

    # Parameters
    ParamsDict = {
        'base_path'               : '.',
        'dem_file_name'           : 'DEM.tif',
        'dsm_file_name'           : 'DSM.tif',
        'ndsm_file_name'          : 'nDSM.tif',
        'ndsm_min_ht'             : 0, # lower limit to remove noise
        'ndsm_max_ht'             : 20, # upper limit to remove noise
        'MaskImage'               : True,
        'Reproject'               : True,
        'StudyAreaShp'            : 'MaskLayer.shp',
        'HistogramEqualizeOutput' : True,
        'CRS'                     : 'EPSG:32619', # EPSG Reference System ID
    }
    # Generate Normalized Digital Surface Model (NDSM)
    ndsmgen = NDSM_Generator_Class(ParamsDict).GetInstance()    
    ndsmgen.GenerateNDSM()                    

class NDSM_Generator_Class():
    """
    Class with functions to generate Normalized Digital Surface Model.
    """

    def __init__(self, Params):
        self.base_path                =  Params['base_path']
        self.dem                      =  Params['dem_file_name']
        self.dsm                      =  Params['dsm_file_name']
        self.ndsm                     =  Params['ndsm_file_name']
        self.ndsm_min_ht              =  Params['ndsm_min_ht']
        self.ndsm_max_ht              =  Params['ndsm_max_ht']
        self.CRS                      =  Params['CRS']
        self.StudyAreaShp             =  Params['StudyAreaShp']
        self.HistogramEqualizeOutput  =  Params['HistogramEqualizeOutput']
        self.MaskImage                =  Params['MaskImage']
        self.Reproject                =  Params['Reproject']

    def ReprojectImage(self):
        """ Repoject the input image \n 
        
        Keyword Arguments: \n 
            self: -- this object
        """
        # Reproject dem
        in_image = os.path.join(self.base_path, 'InputFiles', self.dem)
        out_image = os.path.join(self.base_path, 'OutputFiles', self.dem)
        RasterOperators.ReprojectImage(in_image, out_image, self.CRS)
    
        # Reproject dsm
        in_image = os.path.join(self.base_path, 'InputFiles', self.dsm)
        out_image = os.path.join(self.base_path, 'OutputFiles', self.dsm)
        RasterOperators.ReprojectImage(in_image, out_image, self.CRS)

        print(in_image)

    def CropImageByShpMask(self):
        """ Crop Image by Shape Mask \n 
        
        Keyword Arguments: \n 
            self: -- this object
        """
       
        in_shp_file = os.path.join(self.base_path, 'InputFiles', 'ShapeMask', self.StudyAreaShp)
        # with rasterio.open(InputImage) as ImageObj:
        #     out_image, out_transform = rasterio.mask.mask(ImageObj, gpd.GeoSeries(Polygon(CrownBuffer)), crop=True, filled=True, nodata = 0)

        with fiona.open(in_shp_file, 'r') as shapefile:
            ShapeMask = [feature["geometry"] for feature in shapefile]
            # Crop dem
            RasterOperators.CropImage(os.path.join(self.base_path,'InputFiles',self.dem), ShapeMask, os.path.join(self.base_path,'InputFiles',self.dem))
            # Crop dsm
            RasterOperators.CropImage(os.path.join(self.base_path,'InputFiles',self.dsm), ShapeMask, os.path.join(self.base_path,'InputFiles',self.dsm))

    def BandNormalize(self,band_array):
        """ Normalize Input Image Band \n 
        
        Keyword Arguments: \n 
            self       : -- this object
            band_array : -- input image band
        """
        array_min, array_max = np.amin(band_array), np.amax(band_array)
        return img_as_float32((band_array - array_min) / (array_max - array_min))

    def GenerateNDSM(self):
        """ Generate Normalized Digital Surface Model \n 
        
        Keyword Arguments: \n 
            self: -- this object
        """
        # Reproject data to the the destination CRS.
        if(self.Reproject):
            self.ReprojectImage()
        
        # Crop data using input shape mask.
        if(self.MaskImage):
            self.CropImageByShpMask()

        # Oead the DEM
        with rasterio.open(os.path.join(self.base_path,'InputFiles',self.dem)) as dem_src:
            
            # Open the DSM
                with rasterio.open(os.path.join(self.base_path,'InputFiles',self.dsm)) as dsm_src:
                    meta = dem_src.meta

                    # Open the Output File
                    with rasterio.open(os.path.join(self.base_path,'OutputFiles',self.ndsm), 'w', **meta) as ndsm_dst:
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
                            if(self.HistogramEqualizeOutput):
                                ndsm_band_item = img_as_float32(RasterOperators.HistogramEqualizeImage(ndsm_band_item))
                            
                            # Normalize and Scale
                            # ndsm_band_item = self.BandNormalize(ndsm_band_item)*np.finfo(np.float32).max  # or maxheigt

                            ndsm_dst.write_band(bandId, ndsm_band_item)


    def GetInstance(self):
        return self

GenerateNDSM()
