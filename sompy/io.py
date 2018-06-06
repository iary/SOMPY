#! /usr/bin/env python

import sys,fnmatch
import numpy as np
from .hdf5 import HDF5
from .codebook import Codebook
from .sompy import SOM

    

class OutputToHDF5(HDF5):
    
    def __init__(self,outfile):
        super(self.__class__,self).__init__(outfile,'w')        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return
    
    def __call__(self,SOM):
        self.writeSOM(SOM)
        return

    def writeCodebook(self,codebook):
        self.mkGroup("/codebook")
        attrib = {}
        attrib["lattice"] = codebook.lattice
        attrib["nnodes"] = codebook.nnodes
        attrib["initialized"] = codebook.initialized
        attrib["mapsize"] = codebook.mapsize
        self.addAttributes("/codebook",attrib)                
        self.writeDataset(codebook.matrix,"matrix",hdfdir="/codebook")
        return

    def writeDataset(self,arr,name,hdfdir="/"):
        self.addDataset(hdfdir,name,arr,maxshape=arr.shape)
        return

    def writeSOM(self,SOM,storeNormalizedData=False):                
        # Store basic attributes to dictionary
        attrib = {}
        keys = ["name","training","initialization","_dim","_dlabel","_dlen","mapshape"]
        for key in keys:
            if SOM.__dict__[key] is None:
                attrib[key] = "None"
            else:
                attrib[key] = SOM.__dict__[key]
        # Write name of neighborhood and normalizer objects
        attrib['neighborhood'] = SOM.neighborhood.name
        attrib['_normalizer'] = SOM._normalizer.name
        # Write attributes to file
        self.addAttributes("/",attrib)                
        # Write BMU
        self.writeDataset(SOM._bmu,"_bmu",hdfdir="/")
        # Write mask
        self.writeDataset(SOM.mask,"mask",hdfdir="/")
        # Write _data
        if storeNormalizedData:
            self.writeDataset(SOM._data,"_data",hdfdir="/")
        # Write data_raw
        self.writeDataset(SOM.data_raw,"data_raw",hdfdir="/")
        # Write distance matrix
        self.writeDataset(SOM._distance_matrix,"_distance_matrix",hdfdir="/")
        # Write component names
        self.writeDataset(SOM._component_names,"_component_names",hdfdir="/")
        # Write codebook
        self.writeCodebook(SOM.codebook)
        return
    
    
class InputFromHDF5(HDF5):

    def __init__(self,hdf5File):
        super(self.__class__,self).__init__(hdf5File,'r')
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return

    def __call__(self):
        return self.createSOM()
        

    def createCodebook(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create codebook object
        attrib = self.readAttributes("/codebook")
        CODE = Codebook(list(attrib["mapsize"]),lattice=attrib["lattice"])
        CODE.nnodes = int(attrib["nnodes"])
        CODE.matrix = np.array(self.fileObj["/codebook/matrix"])
        CODE.initialized = attrib["initialized"]
        return CODE        
    
    def createNeighborhood(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Get name for neighborhood object
        name = self.readAttributes("/",required=["neighborhood"])["neighborhood"]
        # Create neighborhood object
        if fnmatch.fnmatch(name,"gaussian"):
            from .neighborhood import GaussianNeighborhood
            NEIGHBORHOOD = GaussianNeighborhood()
        elif fnmatch.fnmatch(name,"bubble"):
            from .neighborhood import BubbleNeighborhood
            NEIGHBORHOOD = BubbleNeighborhood()
        else:
            raise ValueError("ERROR! "+funcname+"(): Neighborhood not recognized! "+\
                                 "Should be 'gaussian' or 'bubble'.")
        return NEIGHBORHOOD
    
    
    def createNormalizer(self):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Get name for normalizer
        name = self.readAttributes("/",required=["_normalizer"])["_normalizer"]
        # Create normalizer object
        if fnmatch.fnmatch(name,"var"):
            from .normalization import VarianceNormalizator
            NORM = VarianceNormalizator()
        elif fnmatch.fnmatch(name,"range"):
            from .normalization import RangeNormalizator
            NORM = RangeNormalizator()
        elif fnmatch.fnmatch(name,"log"):
            from .normalization import LogNormalizator
            NORM = LogNormalizator()
        elif fnmatch.fnmatch(name,"logistic"):
            from .normalization import LogisticNormalizator
            NORM = LogisticNormalizator()
        elif fnmatch.fnmatch(name,"histd"):
            from .normalization import HistDNormalizator
            NORM = HistDNormalizator()
        elif fnmatch.fnmatch(name,"histc"):
            from .normalization import HistCNormalizator
            NORM = HistCNormalizator()
        else:
            raise ValueError("ERROR! "+funcname+"(): Normalization not recognized! "+\
                                 "Should be 'var', 'range', 'log', 'logistic',"+\
                                 " 'histd' or 'histc'.")
        return NORM
        
            
    def createSOM(self):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Read and create codebook object
        CODE = self.createCodebook()
        # Extract raw data
        data = np.array(self.fileObj["/data_raw"])
        # Extract component names
        components = np.array(self.fileObj["/_component_names"])        
        # Create neighborhood and normalizer classes
        neighborhood = self.createNeighborhood()
        normalizer = self.createNormalizer()
        # Read remaining attributes
        attrib = self.readAttributes("/")
        # Create SOM object
        som = SOM(data,neighborhood,normalizer=normalizer,\
                      mapsize=CODE.mapsize,mask=None,mapshape=attrib["mapshape"],\
                      lattice=CODE.lattice,initialization=attrib["initialization"],\
                      training=attrib["training"],name=attrib["name"],\
                      component_names=components)
        # Update attributes
        som.mask = np.array(self.fileObj["/mask"])
        som._dlabel = attrib["_dlabel"]
        som._bmu = np.array(self.fileObj["/_bmu"])
        som.codebook = CODE
        som._distance_matrix = np.array(self.fileObj["/_distance_matrix"])
        return som


