#-------------------------------------------------
#
# Project created by QtCreator 2015-02-14T19:40:22
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = Assignment5
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

INCLUDEPATH += LIBS += "/usr/local/include/"

CONFIG += c++11

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
