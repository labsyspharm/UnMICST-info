from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO, emit
from werkzeug import secure_filename
import time
import os
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from ImageScience import IS, DL
import json
import threading
from toolbox.ftools import *
from toolbox.imtools import tifread, normalize, tifwrite, imshow

MAIN_FOLDER = '/home/mc457/Workspace/FlaskServer' # create manually
DATA_SUBFOLDER = pathjoin(MAIN_FOLDER,'Data') # create manually
ML_SUBFOLDER_ANNOTATIONS = pathjoin(pathjoin(MAIN_FOLDER,'MachineLearning'),'TrainingData') # create manually
ML_SUBFOLDER_MODELS = pathjoin(pathjoin(MAIN_FOLDER,'MachineLearning'),'TrainedModels') # create manually

app = Flask(__name__)
app.config['MAIN_FOLDER'] = MAIN_FOLDER
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app,async_mode='threading')
# https://github.com/miguelgrinberg/Flask-SocketIO/blob/master/example/app.py

# Saved as example, but not used because threads don't work well with heavy processing.
# See https://github.com/miguelgrinberg/Flask-SocketIO/issues/192
#
# def tmls(): # machine learning segmenter (pixel/voxel classifier)
#     IS.mlSegmenterAnnotationsPath = ML_SUBFOLDER_ANNOTATIONS
#     IS.trainMLSegmenter()
#     emit('dialog', 'done training')
# thread = threading.Thread(target=tmls, args=()) # note: if passing a string argument, use comma afterwards; e.g. args=('text',)
# thread.daemon = True
# thread.start()

def shape2String(shape):
    s = '('
    for i in range(len(shape)-1):
        s = s+'%d, ' % shape[i]
    return s+'%d)' % shape[-1]

def pathList2String(l):
    plStr = ''
    for i in range(len(l)-1):
        [p,n,e] = fileparts(l[i])
        idx = '(%d) ' % (i+1)
        plStr = plStr+idx+n+e+', '
    [p,n,e] = fileparts(l[-1])
    idx = '(%d) ' % (len(l))
    plStr = plStr+idx+n+e
    return plStr

def jsonifyCurrentPlane():
    shape = IS.imageShape
    maskType = IS.maskType
    if len(shape) == 2:
        uint8I = np.uint8(255*normalize(IS.I))
        if maskType == 'noMask':
            return jsonify(tcpi=[0,0,0],shape=shape,data=uint8I.tolist(),planeView=IS.currentViewPlane,mask=[],maskType='noMask')
        if maskType == 'labelMask':
            mask = IS.labelMask
        elif maskType == 'segmMask':
            mask = IS.segmMask
        return jsonify(tcpi=[0,0,0],shape=shape,data=uint8I.tolist(),planeView=IS.currentViewPlane,mask=mask.tolist(),maskType=maskType)            
    if len(shape) == 3: # plane, row, col
        uint8I = np.uint8(255*normalize(IS.I[IS.planeIndex,:,:]))
        if maskType == 'noMask':
            return jsonify(tcpi=[0,0,IS.planeIndex],shape=shape,data=uint8I.tolist(),planeView=IS.currentViewPlane,mask=[],maskType='noMask')
        if maskType == 'labelMask':
            mask = IS.labelMask[IS.planeIndex,:,:]
        elif maskType == 'segmMask':
            mask = IS.segmMask[IS.planeIndex,:,:]
        return jsonify(tcpi=[0,0,IS.planeIndex],shape=shape,data=uint8I.tolist(),planeView=IS.currentViewPlane,mask=mask.tolist(),maskType=maskType)
    if len(shape) == 4: # plane, channel, row, col
        uint8I = np.uint8(255*normalize(IS.I[IS.planeIndex,IS.channelIndex,:,:]))
        if maskType == 'noMask':
            return jsonify(tcpi=[0,IS.channelIndex,IS.planeIndex],shape=shape,data=uint8I.tolist(),planeView=IS.currentViewPlane,mask=[],maskType='noMask')
        if maskType == 'labelMask':
            mask = IS.labelMask[IS.planeIndex,:,:]
        elif maskType == 'segmMask':
            mask = IS.segmMask[IS.planeIndex,:,:]
        return jsonify(tcpi=[0,IS.channelIndex,IS.planeIndex],shape=shape,data=uint8I.tolist(),planeView=IS.currentViewPlane,mask=mask.tolist(),maskType=maskType)

def ipDialogStep1(ipOp, followUpQuestion):
    if IS.hasImage:
        DL.ipOperation = ipOp
        DL.ipStep = 1
        emit('dialog', followUpQuestion)
        emit('server2ClientMessage', 'animate last dialog message')
    else:
        emit('dialog', 'set image first')

def extractPlaneDialog():
    if IS.hasImage and IS.nPlanes > 0 and IS.nChannels == 0:
        DL.ipOperation = 'extract plane'
        DL.ipStep = 1
        emit('dialog', 'which plane? [1,...,%d]' % IS.nPlanes)
        emit('server2ClientMessage', 'animate last dialog message')
    else:
        emit('dialog', 'operation invalid for this image')

def extractChannelDialog():
    if IS.hasImage and IS.nChannels > 0:
        DL.ipOperation = 'extract channel'
        DL.ipStep = 1
        emit('dialog', 'which channel? [1,...,%d]' % IS.nChannels)
        emit('server2ClientMessage', 'animate last dialog message')
    else:
        emit('dialog', 'operation invalid for this image')

# 'connect' messages are test messages
# 'server2ClientMessage' messages are hidden to the user
# 'dialog' messages are shown on the dialog area

@socketio.on('connect')                                                         
def connect():             
    print('server side connected')
    emit('server2ClientMessage', 'Server: SocketIO connected')

@socketio.on('client2ServerMessage')
def client2ServerMessage(message):
    if message == 'socket echo test':
        emit('server2ClientMessage', message)
    elif message == 'initialize':
        IS.reset()
        DL.reset()
        emit('server2ClientMessage', 'server initialized')
        l = listfiles(ML_SUBFOLDER_ANNOTATIONS,'.tif')
        for i in range(len(l)):
            removeFile(l[i])
    elif message == 'create label mask':
        IS.maskType = 'labelMask'
        if IS.labelMask is None: # new annotation from scratch
            IS.createLabelMask()
        else: # editing annotation, thus just tell client to show label mask
            emit('server2ClientMessage', 'fetch plane')
        emit('server2ClientMessage', 'label mask created')

    elif message == 'save label mask':
        IS.saveLabelMask(ML_SUBFOLDER_ANNOTATIONS)
        IS.maskType = 'noMask'
        emit('server2ClientMessage', 'done saving annotations')
    elif message == 'train ml segmenter':
        IS.mlSegmenterAnnotationsPath = ML_SUBFOLDER_ANNOTATIONS
        IS.trainMLSegmenter()
        emit('dialog', 'done training')
    elif message[:10] == 'view plane':
        if IS.hasImage and len(IS.imageShape) == 3:
            newViewPlane = message[11]
            # print(IS.currentViewPlane,'->',newViewPlane)
            IS.imageProcessing('view plane',{'plane': newViewPlane, 'currentViewPlane': IS.currentViewPlane})
            IS.currentViewPlane = newViewPlane
            emit('server2ClientMessage', 'fetch plane')
            # emit('dialog', 'operation %s done' % message)
        else:
            emit('dialog', 'operation available only for 3D images')
    elif message[:13] == 'set mask type':
        IS.maskType = message[14:]
        print('new mask type: ', IS.maskType)
    else:
        print('client2ServerMessage', message)

@socketio.on('dialog')
def dialog(message):
    if message == 'image properties':
        if not IS.hasImage:
            emit('dialog', 'image not set | enter \'new image\' to set new image')
        else:
            emit('dialog', 'shape: %s | min: %f | max: %f' % (shape2String(IS.imageShape), np.min(IS.I), np.max(IS.I)))
    elif message == 'new image from server':
        DL.newImageFromServerStep = 1
        IS.imagesOnServer = listfiles(DATA_SUBFOLDER,'.tif')
        lstr = 'choose image: '+pathList2String(IS.imagesOnServer)
        emit('dialog', lstr)
        emit('server2ClientMessage', 'animate last dialog message')
    elif message == 'reset image': # reset to original image (before processing)
        if IS.hasImage:
            IS.setImage(IS.originalImage)
            if len(IS.imageShape) == 3:
                IS.currentViewPlane = 'z'
            else:
                IS.currentViewPlane = ''
            IS.maskType = 'noMask'
            IS.labelMask = None
            emit('server2ClientMessage', 'fetch plane')
            emit('dialog', 'shape: %s | min: %f | max: %f' % (shape2String(IS.imageShape), np.min(IS.I), np.max(IS.I)))
        else:
            emit('server2ClientMessage', 'no image to set')
    elif message == 'save current image to server':
        if IS.hasImage:
            DL.saveCurrentImageToServerStep = 1
            emit('dialog', 'enter image name | first character must be a letter | all characters should be alphanumeric | do not include extension (such as .tif)')
            emit('server2ClientMessage', 'animate last dialog message')
        else:
            emit('dialog', 'no image to save')
    elif message == 'save annotations to server':
        IS.unsavedAnnotationsOnServer = listfiles(ML_SUBFOLDER_ANNOTATIONS,'.tif')
        if IS.unsavedAnnotationsOnServer:
            DL.saveAnnotationsToServerStep = 1
            emit('dialog', 'enter folder name to save annotations in | first character must be a letter | all characters should be alphanumeric')
            emit('server2ClientMessage', 'animate last dialog message')
        else:
            emit('dialog', 'no annotations to save')
    elif message == 'load annotations from server':
        print('load annotations from server')
        IS.annotationSetsOnServer = listsubdirs(ML_SUBFOLDER_ANNOTATIONS)
        if IS.annotationSetsOnServer:
            DL.loadAnnotationsFromServerStep = 1
            lstr = 'choose annotation set: '+pathList2String(IS.annotationSetsOnServer)
            emit('dialog', lstr)
            emit('server2ClientMessage', 'animate last dialog message')
        else:
            emit('dialog', 'no annotation sets available')
    elif message == 'edit annotations':
        IS.unsavedAnnotationsOnServer = listfiles(ML_SUBFOLDER_ANNOTATIONS,'.tif')
        if IS.unsavedAnnotationsOnServer:
            DL.editAnnotationsStep = 1
            emit('dialog', 'enter index of annotations to edit [1,...,%d]' % (len(IS.unsavedAnnotationsOnServer)/2))
            emit('server2ClientMessage', 'animate last dialog message')
        else:
            emit('dialog', 'no annotations to edit')
    elif message == 'save ml model to server':
        if IS.mlSegmenterModel:
            DL.saveMLModelToServerStep = 1
            emit('dialog', 'enter model name | first character must be a letter | all characters should be alphanumeric | do not include extension (such as .txt)')
            emit('server2ClientMessage', 'animate last dialog message')
        else:
            emit('dialog', 'no model to save')
    elif message == 'load ml model from server':
        IS.mlModelsOnServer = listfiles(pathjoin(MAIN_FOLDER,ML_SUBFOLDER_MODELS),'.obj')
        l = IS.mlModelsOnServer
        if len(l) > 0:
            DL.loadMLModelFromServerStep = 1
            lstr = 'choose model: '+pathList2String(l)
            emit('dialog', lstr)
            emit('server2ClientMessage', 'animate last dialog message')
        else:
            emit('dialog', 'none available')
    elif message == 'ml probability maps':
        if IS.hasImage:
            if IS.mlSegmenterModel:
                emit('dialog','computing...')
                IS.imageProcessing('ml probability maps',{})
                emit('server2ClientMessage', 'fetch plane')
                emit('dialog', 'operation '+message+' done')
            else:
                emit('dialog', 'no ml model available')
        else:
            emit('dialog', 'set image first')
    elif message == 'median filter':
        ipDialogStep1('median filter', 'which filter radius? [2,3,...]')
    elif message == 'maximum filter':
        ipDialogStep1('maximum filter', 'which filter radius? [2,3,...]')
    elif message == 'minimum filter':
        ipDialogStep1('minimum filter', 'which filter radius? [2,3,...]')
    elif message == 'blur':
        ipDialogStep1('blur', 'which scale? [1,2,3,...]')
    elif message == 'log':
        ipDialogStep1('log', 'which scale? [1,2,3,...]')
    elif message == 'gradient magnitude':
        ipDialogStep1('gradient magnitude', 'which scale? [1,2,3,...]')
    elif message == 'derivatives':
        ipDialogStep1('derivatives', 'which scale? [1,2,3,...]')
    elif message == 'extract plane':
        extractPlaneDialog()
    elif message == 'extract channel':
        extractChannelDialog()
    elif DL.newImageFromServerStep == 1:
        if message.isnumeric():
            idx = int(message)-1
            if idx >= 0 and idx < len(IS.imagesOnServer):
                IS.setImage(tifread(IS.imagesOnServer[idx]))
                IS.originalImage = IS.I
                if len(IS.imageShape) == 3:
                    IS.currentViewPlane = 'z'
                else:
                    IS.currentViewPlane = ''
                IS.maskType = 'noMask'
                IS.labelMask = None
                DL.newImageFromServerStep = 0
                emit('server2ClientMessage', 'fetch plane');
                emit('dialog', 'shape: %s | min: %f | max: %f' % (shape2String(IS.imageShape), np.min(IS.I), np.max(IS.I)))
            else:
                emit('dialog','index out of bounds')
        else:
            emit('dialog','please enter one of the provided indices')
    elif DL.saveCurrentImageToServerStep == 1:
        if len(message) > 0:
            if message[0].isalpha():
                if message.isalnum():
                    path = pathjoin(DATA_SUBFOLDER, '%s.tif' % message)
                    tifwrite(np.uint8(255*normalize(IS.I)),path)
                    emit('dialog', 'saved to server as %s.tif' % message)
                    DL.saveCurrentImageToServerStep = 0
                else:
                    emit('dialog', 'not all chars are alphanumeric')
            else:
                emit('dialog','first char should be a letter')
        else:
            emit('dialog', 'name should have at least one char')
    elif DL.saveAnnotationsToServerStep == 1:
        if len(message) > 0:
            if message[0].isalpha():
                if message.isalnum():
                    path = pathjoin(ML_SUBFOLDER_ANNOTATIONS, '%s' % message)
                    IS.saveAnnotations(path)
                    emit('dialog', 'annotations saved under %s' % message)
                    DL.saveAnnotationsToServerStep = 0
                else:
                    emit('dialog', 'not all chars are alphanumeric')
            else:
                emit('dialog','first char should be a letter')
        else:
            emit('dialog', 'name should have at least one char')
    elif DL.loadAnnotationsFromServerStep == 1:
        if len(message) > 0 and message.isnumeric():
            idx = int(message)
            if idx >= 1 and idx <= len(IS.annotationSetsOnServer):
                IS.loadAnnotations(IS.annotationSetsOnServer[idx-1])
                [p,n,e] = fileparts(IS.annotationSetsOnServer[idx-1])
                emit('dialog', 'annotations '+n+' loaded')
                emit('server2ClientMessage', 'did annotate images')
                DL.loadAnnotationsFromServerStep = 0
            else:
                emit('dialog', 'invalid index')
        else:
            emit('dialog', 'invalid input')
    elif DL.editAnnotationsStep == 1:
        if len(message) > 0 and message.isnumeric():
            idx = int(message)
            if idx >= 1 and idx <= (len(IS.unsavedAnnotationsOnServer)/2):
                I = tifread(pathjoin(ML_SUBFOLDER_ANNOTATIONS, 'Image%02d_Img.tif' % (idx-1)))
                A = tifread(pathjoin(ML_SUBFOLDER_ANNOTATIONS, 'Image%02d_Ant.tif' % (idx-1)))
                IS.setImage(I)
                IS.originalImage = IS.I
                if len(IS.imageShape) == 3:
                    IS.currentViewPlane = 'z'
                else:
                    IS.currentViewPlane = ''
                IS.maskType = 'noMask' # will be corrected once client loads annotation tool
                IS.labelMask = A
                IS.labelMaskIndex = idx-1
                DL.editAnnotationsStep = 0
                emit('server2ClientMessage', 'fetch plane');
                emit('dialog', 'shape: %s | min: %f | max: %f' % (shape2String(IS.imageShape), np.min(IS.I), np.max(IS.I)))
                emit('server2ClientMessage', 'set_nClasses%d' % np.max(A))
                emit('server2ClientMessage', 'load annotation tool')
            else:
                emit('dialog', 'invalid index')
        else:
            emit('dialog', 'invalid input')
    elif DL.saveMLModelToServerStep == 1:
        if len(message) > 0:
            if message[0].isalpha():
                if message.isalnum():
                    path = pathjoin(ML_SUBFOLDER_MODELS, '%s.obj' % message)
                    IS.saveMLModel(path)
                    emit('dialog', 'model saved as %s.obj' % message)
                    DL.saveMLModelToServerStep = 0
                else:
                    emit('dialog', 'not all chars are alphanumeric')
            else:
                emit('dialog','first char should be a letter')
        else:
            emit('dialog', 'name should have at least one char')
    elif DL.loadMLModelFromServerStep == 1:
        if len(message) > 0 and message.isnumeric():
            idx = int(message)
            if idx >= 1 and idx <= len(IS.mlModelsOnServer):
                IS.loadMLModel(IS.mlModelsOnServer[idx-1])
                emit('dialog', 'model loaded')
                emit('server2ClientMessage', 'did train ml segmenter')
                emit('server2ClientMessage', 'set_nClasses%d' % IS.mlSegmenterModel['nClasses'])
                DL.loadMLModelFromServerStep = 0
            else:
                emit('dialog', 'invalid index')
        else:
            emit('dialog', 'invalid input')
    elif DL.ipStep == 1: # image processing step
        if len(message) > 0 and message.isnumeric():
            didSucceed = True
            prm = int(message)
            if any(DL.ipOperation == ipOp for ipOp in ['median filter','maximum filter','minimum filter']):
                emit('dialog','computing...')
                IS.imageProcessing(DL.ipOperation,{'size': prm})
            elif any(DL.ipOperation == ipOp for ipOp in ['blur','log','gradient magnitude','derivatives']):
                emit('dialog','computing...')
                IS.imageProcessing(DL.ipOperation,{'sigma': prm})
            elif DL.ipOperation == 'extract plane':
                if prm >= 1 and prm <= IS.nPlanes:
                    IS.imageProcessing(DL.ipOperation,{'plane': prm})
                else:
                    didSucceed = False
                    emit('dialog', 'index out of bounds')
            elif DL.ipOperation == 'extract channel':
                if prm >= 1 and prm <= IS.nChannels:
                    IS.imageProcessing(DL.ipOperation,{'channel': prm})
                else:
                    didSucceed = False
                    emit('dialog', 'index out of bounds')
            if didSucceed:
                DL.ipStep = 0
                emit('server2ClientMessage', 'fetch plane')
                emit('dialog', 'operation '+DL.ipOperation+' done')
        else:
            emit('dialog', 'invalid input')
    else:
        emit('dialog', 'invalid input')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/jsonecho')
def jsonEcho():
    message = request.args.get('message','',type=str) # id, defaults, type
    return jsonify(message=message)

@app.route('/jsonupload', methods=['POST'])
def jsonUpload():
    data = request.form['text']
    parsed = json.loads(data)
    if parsed['description'] == 'annotation':
        IS.updateLabelMask(parsed['content'])
        if len(IS.imageShape) == 2:
            IS.saveLabelMask(ML_SUBFOLDER_ANNOTATIONS)
            IS.maskType = 'noMask'
        return jsonify(message='annotations uploaded')
    elif parsed['description'] == 'mlsegtrainprmts': # ML segmentation training parameters
        IS.mlSegmenterTrainParameters = parsed['content']
        # training procedure actually started by client after receiving following message
        return jsonify(message='training ml segmenter | this might take a while | a message will be shown here when training is done')
    elif parsed['description'] == 'mlsegprmts': # ml segmentation parameters
        mlsegprmts = parsed['content']
        pmIdx = mlsegprmts[0]-1
        segBlr,segThr = mlsegprmts[1:3]
        IS.mlSegment(pmIdx,segBlr/100,segThr/100)
        IS.maskType = 'segmMask'
        return jsonifyCurrentPlane()
    elif parsed['description'] == 'thrsegprmts': # threshold segmentation parameters
        thrsegprmts = parsed['content']
        cIdx = thrsegprmts[0]-1 # 0: dark, 1: bright
        segBlr,segThr = thrsegprmts[1:3]
        IS.thresholdSegment(cIdx,segBlr/100,segThr/100)
        IS.maskType = 'segmMask'
        return jsonifyCurrentPlane()
    elif parsed['description'] == 'plnfrmsvr': # plane from server; content: [time,channel,plane]
        IS.planeIndex = parsed['content'][2]
        IS.channelIndex = parsed['content'][1]
        # time index not supported yet
        return jsonifyCurrentPlane()
    elif parsed['description'] == 'fetchpln':
        return jsonifyCurrentPlane()

        
@app.route('/jsonfileupload', methods=['POST'])
def jsonFileUpload():
    if 'file' not in request.files:
        print('No file part')
    file = request.files['file']
    if file.filename == '':
        print('No selected file')
    I = tifread(file)
    IS.setImage(I)
    IS.originalImage = IS.I
    IS.currentViewPlane = '' # this only sets 2D images
    IS.maskType = 'noMask'
    IS.labelMask = None
    shape = list(I.shape)
    uint8I = np.uint8(255*IS.I)
    emit('dialog', 'shape: %s | min: %f | max: %f' % (shape2String(IS.imageShape), np.min(IS.I), np.max(IS.I)))
    return jsonify(tcpi=[0,0,0],shape=shape,data=uint8I.tolist(),planeView=IS.currentViewPlane,mask=[],maskType='noMask')

if __name__ == "__main__":
    # deploys to the same machine (accessible at localhost:5000)
    # socketio.run(app, debug=True)

    # deploys to local wifi network
    # client should point to IP address of server in local network
    # ip can be found on Settings (Ubuntu) or System Preferences (Mac) app
    # e.g.: accessible at 10.0.0.104:5000
    socketio.run(app, debug=False, host='0.0.0.0')
    