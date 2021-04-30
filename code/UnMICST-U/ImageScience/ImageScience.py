import numpy as np
from toolbox import pixelclassifier as pc
from toolbox import voxelclassifier as vc
from toolbox.imtools import *
from toolbox.ftools import *
import pickle

# -------------------------

class ImageProcessor:
    def __init__(self,**kwargs):
        self.args = kwargs

class IPBlur(ImageProcessor):
    def run(self,I):
        return imgaussfilt(I,self.args['sigma'])

class IPLoG(ImageProcessor):
    def run(self,I):
        return imlogfilt(I,self.args['sigma'])

class IPGradMag(ImageProcessor):
    def run(self,I):
        return imgradmag(I,self.args['sigma'])

class IPDerivatives(ImageProcessor):
    def run(self,I):
        if len(I.shape) == 2:
            return np.moveaxis(imderivatives(I,self.args['sigma']),[2,0,1],[0,1,2])
        if len(I.shape) == 3:
            return np.moveaxis(imderivatives3(I,self.args['sigma']),[0,3,1,2],[0,1,2,3])
        if len(I.shape) > 3:
            print('case len(I.shape) > 3 not considered')

class IPThresholdSegment(ImageProcessor):
    def run(self,I):
        C = I
        if self.args['pcIdx'] == 0: # dark pixels
            C = 1-C
        M = thrsegment(C,self.args['segBlr'],self.args['segThr'])
        return M.astype(int)

class IPPixelClassifierSegment(ImageProcessor):
    def run(self,I):
        C = pc.classify(I,self.args['model'],output='probmaps')
        M = thrsegment(C[:,:,self.args['pmIdx']],self.args['segBlr'],self.args['segThr'])
        return M.astype(int)        

class IPVoxelClassifierSegment(ImageProcessor):
    def run(self,I):
        C = vc.classify(I,self.args['model'],output='probmaps')
        M = thrsegment(C[:,:,:,self.args['pmIdx']],self.args['segBlr'],self.args['segThr'])
        return M.astype(int)

class IPMachLearnProbMaps(ImageProcessor):
    def run(self,I):
        if len(I.shape) == 2:
            return np.moveaxis(pc.classify(I,self.args['model'],output='probmaps'),[2,0,1],[0,1,2])
        if len(I.shape) == 3:
            return np.moveaxis(vc.classify(I,self.args['model'],output='probmaps'),[0,3,1,2],[0,1,2,3])
        if len(I.shape) > 3:
            print('case len(I.shape) > 3 not considered')

class IPExtractPlane(ImageProcessor):
    def run(self,I):
        if len(I.shape) == 3:
            return I[self.args['index'],:,:]
        else:
            print('invalid tensor dimension')

class IPExtractChannel(ImageProcessor):
    def run(self,I):
        if len(I.shape) == 4:
            return I[:,self.args['index'],:,:]
        else:
            print('invalid tensor dimension')

class IPMedianFilter(ImageProcessor):
    def run(self,I):
        return medfilt(I,self.args['size'])

class IPMaximumFilter(ImageProcessor):
    def run(self,I):
        return maxfilt(I,self.args['size'])

class IPMinimumFilter(ImageProcessor):
    def run(self,I):
        return minfilt(I,self.args['size'])

class IPViewPlane(ImageProcessor):
    def run(self, I):
        if len(I.shape) == 3:
            if self.args['currentViewPlane'] == 'z':
                if self.args['plane'] == 'x':
                    return np.moveaxis(I,[2,0,1],[0,1,2])
                if self.args['plane'] == 'y':
                    return np.moveaxis(I,[1,2,0],[0,1,2])
                else:
                    return I
            if self.args['currentViewPlane'] == 'y':
                if self.args['plane'] == 'z':
                    return np.moveaxis(I,[2,0,1],[0,1,2])
                if self.args['plane'] == 'x':
                    return np.moveaxis(I,[1,2,0],[0,1,2])
                else:
                    return I
            if self.args['currentViewPlane'] == 'x':
                if self.args['plane'] == 'y':
                    return np.moveaxis(I,[2,0,1],[0,1,2])
                if self.args['plane'] == 'z':
                    return np.moveaxis(I,[1,2,0],[0,1,2])
                else:
                    return I
        else:
            print('IPViewPlane only applicable to 3D images')



# -------------------------

class DL: # dialog logic
    newImageFromServerStep = 0
    saveCurrentImageToServerStep = 0
    saveAnnotationsToServerStep = 0
    loadAnnotationsFromServerStep = 0
    editAnnotationsStep = 0
    saveMLModelToServerStep = 0
    loadMLModelFromServerStep = 0
    ipOperation = None
    ipParameters = None
    ipStep = 0

    def reset(): # called when browser reloads
        DL.newImageFromServerStep = 0
        DL.saveCurrentImageToServerStep = 0
        DL.saveAnnotationsToServerStep = 0
        DL.loadAnnotationsFromServerStep = 0
        DL.editAnnotationsStep = 0
        DL.saveMLModelToServerStep = 0
        DL.loadMLModelFromServerStep = 0
        DL.ipOperation = None
        DL.ipParameters = None
        DL.ipStep = 0

# -------------------------

class IS: # image science
    I = None # current image, fetched by client along with labelmask or segmMask if present
    labelMask = None
    labelMaskIndex = None
    segmMask = None
    maskType = None
    imagesOnServer = None # list of paths to images on server
    unsavedAnnotationsOnServer = None # list of paths to annotations on server not yet saved (to subfolders)
    mlModelsOnServer = None # list of paths to ML models on server
    annotationSetsOnServer = None # list of folders containing annotations
    hasImage = False
    imageType = None
    imageShape = None
    planeIndex = None
    channelIndex = None
    mlSegmenterTrainParameters = None
    mlSegmenterModel = None
    mlSegmenterAnnotationsPath = None
    originalImage = None
    nPlanes = 0
    nChannels = 0
    currentViewPlane = None

    def reset(): # called when browser reloads
        IS.I = None
        IS.labelMask = None
        IS.segmMask = None
        IS.maskType = None # 'noMask', 'labelMask', 'segmMask'
        IS.imagesOnServer = None
        IS.unsavedAnnotationsOnServer = None
        IS.mlModelsOnServer = None
        IS.annotationSetsOnServer
        IS.hasImage = False
        IS.imageType = None
        IS.imageShape = None
        IS.planeIndex = None
        IS.channelIndex = None
        IS.mlSegmenterTrainParameters = None
        IS.mlSegmenterModel = None
        IS.mlSegmenterAnnotationsPath = None
        IS.originalImage = None
        IS.nPlanes = 0
        IS.nChannels = 0
        IS.currentViewPlane = None

    def setImage(Image):
        # currentPlaneView, originalImage, maskType set from outside
        # this method can be called to update an existing image (e.g. via image processing)
        IS.I = im2double(Image)
        IS.hasImage = True
        IS.imageShape = Image.shape
        IS.planeIndex = None
        IS.channelIndex = None
        IS.nPlanes = 0
        IS.nChannels = 0
        if len(IS.imageShape) > 2:
            IS.planeIndex = int(IS.imageShape[0]/2)
            IS.nPlanes = IS.imageShape[0]
        if len(IS.imageShape) > 3:
            IS.channelIndex = 0
            IS.nChannels = IS.imageShape[1]

    def createLabelMask():
        IS.labelMaskIndex = None
        if len(IS.imageShape) < 4:
            IS.labelMask = np.uint8(np.zeros(IS.imageShape))
        elif len(IS.imageShape) == 4: # 3D annotation volume
            IS.labelMask = np.uint8(np.zeros((IS.imageShape[0],IS.imageShape[2],IS.imageShape[3])))
        else:
            print('5D image case not considered')

    def updateLabelMask(ant):
        if len(IS.imageShape) == 2:
            IS.labelMask[:] = 0
            for i in range(len(ant)):
                indices = ant[i]
                for j in range(len(indices)):
                    row = int(indices[j]/IS.imageShape[1])
                    col = int(indices[j]-row*IS.imageShape[1])
                    IS.labelMask[row,col] = i+1
        elif len(IS.imageShape) == 3 or len(IS.imageShape) == 4:
            IS.labelMask[IS.planeIndex,:,:] = 0
            for i in range(len(ant)):
                indices = ant[i]
                for j in range(len(indices)):
                    row = int(indices[j]/IS.imageShape[-1])
                    col = int(indices[j]-row*IS.imageShape[-1])
                    IS.labelMask[IS.planeIndex,row,col] = i+1

    def thresholdSegment(pcIdx,segBlr,segThr): # works for 2D and 3D images
        ip = IPThresholdSegment(pcIdx=pcIdx,segBlr=segBlr,segThr=segThr)
        IS.segmMask = ip.run(IS.I)

    def saveLabelMask(path):
        if len(IS.labelMask.shape) == 2:
            I2Save = IS.I
            L2Save = IS.labelMask
        elif len(IS.labelMask.shape) == 3:
            ip = IPViewPlane(plane='z',currentViewPlane=IS.currentViewPlane)
            I2Save = ip.run(IS.I)
            L2Save = ip.run(IS.labelMask)
        else:
            print('saveLabelMask case dimension > 3 not considered')

        if IS.labelMaskIndex is None:
            idx = len(listfiles(path,'_Img.tif'))
        else:
            idx = IS.labelMaskIndex

        tifwrite(L2Save, pathjoin(path, 'Image%02d_Ant.tif' % idx))
        tifwrite(I2Save, pathjoin(path, 'Image%02d_Img.tif' % idx))

        # print('saved annotations with index %d' % idx)
        # print(pathjoin(path, 'Image%02d_Ant.tif' % idx))

    def trainMLSegmenter():
        sigmaDeriv = IS.mlSegmenterTrainParameters[0]
        sigmaLoG = IS.mlSegmenterTrainParameters[1]

        imSh = tifread(listfiles(IS.mlSegmenterAnnotationsPath,'_Img.tif')[0]).shape

        if len(imSh) == 2:
            print('training pixel classifier')
            IS.mlSegmenterModel = pc.train(IS.mlSegmenterAnnotationsPath,sigmaDeriv=sigmaDeriv,sigmaLoG=sigmaLoG,locStatsRad=0)
            # pc.plotFeatImport(IS.mlSegmenterModel['featImport'],IS.mlSegmenterModel['featNames'])
        elif len(imSh) == 3:
            print('training voxel classifier')
            IS.mlSegmenterModel = vc.train(IS.mlSegmenterAnnotationsPath,sigmaDeriv=sigmaDeriv,sigmaLoG=sigmaLoG,locStatsRad=0)
            # pc.plotFeatImport(IS.mlSegmenterModel['featImport'],IS.mlSegmenterModel['featNames'])
        # l = listfiles(IS.mlSegmenterAnnotationsPath, '.tif')
        # [os.remove(l[i]) for i in range(len(l))]

    def mlSegment(pmIdx,segBlr,segThr):
        if len(IS.imageShape) == 2:
            ip = IPPixelClassifierSegment(model=IS.mlSegmenterModel,pmIdx=pmIdx,segBlr=segBlr,segThr=segThr)
        elif len(IS.imageShape) == 3:
            ip = IPVoxelClassifierSegment(model=IS.mlSegmenterModel,pmIdx=pmIdx,segBlr=segBlr,segThr=segThr)
        IS.segmMask = ip.run(IS.I)

    def saveAnnotations(path):
        print('saving annotations')
        createFolderIfNonExistent(path)
        for i in range(len(IS.unsavedAnnotationsOnServer)):
            moveFile(IS.unsavedAnnotationsOnServer[i],path)
    def loadAnnotations(path):
        print('load annotations from', path)
        [p,n,e] = fileparts(path)
        l = listfiles(p,'.tif')
        for i in range(len(l)):
            removeFile(l[i])
        l = listfiles(path,'.tif')
        for i in range(len(l)):
            copyFile(l[i],p)

    def saveMLModel(path):
        print('writing ml model')
        modelFile = open(path, 'wb')
        pickle.dump(IS.mlSegmenterModel, modelFile)

    def loadMLModel(path):
        print('loading ml model')
        modelFile = open(path, 'rb')
        IS.mlSegmenterModel = pickle.load(modelFile)

    def imageProcessing(ipOperation,ipParameters):
        if ipOperation == 'blur':
            ip = IPBlur(sigma=ipParameters['sigma'])
        elif ipOperation == 'log':
            ip = IPLoG(sigma=ipParameters['sigma'])
        elif ipOperation == 'gradient magnitude':
            ip = IPGradMag(sigma=ipParameters['sigma'])
        elif ipOperation == 'derivatives':
            ip = IPDerivatives(sigma=ipParameters['sigma'])
        elif ipOperation == 'ml probability maps':
            ip = IPMachLearnProbMaps(model=IS.mlSegmenterModel)
        elif ipOperation == 'extract plane':
            ip = IPExtractPlane(index=ipParameters['plane']-1)
        elif ipOperation == 'extract channel':
            ip = IPExtractChannel(index=ipParameters['channel']-1)
        elif ipOperation == 'median filter':
            ip = IPMedianFilter(size=ipParameters['size'])
        elif ipOperation == 'maximum filter':
            ip = IPMaximumFilter(size=ipParameters['size'])
        elif ipOperation == 'minimum filter':
            ip = IPMinimumFilter(size=ipParameters['size'])
        elif ipOperation == 'view plane':
            ip = IPViewPlane(plane=ipParameters['plane'],currentViewPlane=ipParameters['currentViewPlane'])
        IS.setImage(ip.run(IS.I))
        if len(IS.imageShape) > 2:
            IS.currentViewPlane = 'z'
        
        if ipOperation == 'view plane':
            if IS.maskType == 'segmMask':
                IS.segmMask = ip.run(IS.segmMask)
            elif IS.maskType == 'labelMask':
                IS.labelMask = ip.run(IS.labelMask)
