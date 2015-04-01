#-------------------------------------------------
#
# Project created by QtCreator 2015-03-26T15:48:09
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = FaceRecognitionProject
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app
CONFIG += c++11


INCLUDEPATH += LIBS += "/usr/local/include/"

debug{

LIBS += "/usr/local/lib/libopencv_core.so.2.4.10"
LIBS += "/usr/local/lib/libopencv_features2d.so.2.4.10"
LIBS += "/usr/local/lib/libopencv_flann.so.2.4.10"
LIBS += "/usr/local/lib/libopencv_gpu.so.2.4.10"
LIBS += "/usr/local/lib/libopencv_highgui.so.2.4.10"
LIBS += "/usr/local/lib/libopencv_imgproc.so.2.4.10"
LIBS += "/usr/local/lib/libopencv_legacy.so.2.4.10"
LIBS += "/usr/local/lib/libopencv_ml.so.2.4.10"
LIBS += "/usr/local/lib/libopencv_objdetect.so.2.4.10"
LIBS += "/usr/local/lib/libopencv_ocl.so.2.4.10"
LIBS += "/usr/local/lib/libopencv_photo.so.2.4.10"
LIBS += "/usr/local/lib/libopencv_stitching.so.2.4.2"
LIBS += "/usr/local/lib/libopencv_ts.so.2.4.2"
LIBS += "/usr/local/lib/libopencv_video.so.2.4.10"
LIBS += "/usr/local/lib/libopencv_videostab.so.2.4.2"
LIBS += "/usr/local/lib/libopencv_calib3d.so.2.4.10"
LIBS += "/usr/local/lib/libopencv_nonfree.so.2.4.2"


}


SOURCES += main.cpp
