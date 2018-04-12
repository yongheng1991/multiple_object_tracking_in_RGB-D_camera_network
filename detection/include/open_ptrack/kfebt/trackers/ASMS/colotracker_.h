#ifndef COLOTRACKER__H
#define COLOTRACKER__H

#include "cv.h"
#include "highgui.h"
#include "region_.h"
#include "histogram_.h"
#include <iostream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/video/video.hpp"

//#define SHOWDEBUGWIN

#define BIN_1 16
#define BIN_2 16
#define BIN_3 16
#define UPDATE_RATE 0.00

class ColorTrackert
{
private:
    cv::Mat im1;
    cv::Mat im2;
    cv::Mat im3;

    cv::Mat im1_old;
    cv::Mat im2_old;
    cv::Mat im3_old;

    Histogramt q_hist;
    Histogramt q_orig_hist;
    Histogramt b_hist;
    Histogramt y1hist;

    double defaultWidth;
    double defaultHeight;

    double wAvgBg;
    double bound1;
    double bound2;

    cv::Point histMeanShift(double x1, double y1, double x2, double y2);
    cv::Point histMeanShiftIsotropicScale(double x1, double y1, double x2, double y2, double * scale, int * iter, double* similarity);

    void preprocessImage(cv::Mat & img);
    void extractBackgroundHistogram(int x1, int y1, int x2, int y2, Histogramt &hist);
    void extractForegroundHistogram(int x1, int y1, int x2, int y2, Histogramt &hist);

    inline double kernelProfile_Epanechnikov(double x)
    { return (x <= 1) ? (2.0/3.14)*(1-x) : 0; }
    //{ return (x <= 1) ? (1-x) : 0; }
    inline double kernelProfile_EpanechnikovDeriv(double x)
    { return (x <= 1) ? (-2.0/3.14) : 0; }
    //{ return (x <= 1) ? -1 : 0; }
public:
    BBoxt lastPosition;
    int frame;
    int sumIter;

    // Init methods
    void init(cv::Mat & img, int x1, int y1, int x2, int y2);

    // Set last object position - starting position for next tracking step
    inline void setLastBBox(int x1, int y1, int x2, int y2)
    {
        lastPosition.setBBox(x1, y1, x2-x1, y2-y1, 1, 1);
    }

    inline BBoxt * getBBox()
    {
        BBoxt * bbox = new BBoxt();
        *bbox = lastPosition;
        return bbox;
    }

    // frame-to-frame object tracking
    BBoxt * track(cv::Mat & img, double x1, double y1, double x2, double y2, double* confidence);
    inline BBoxt * track(cv::Mat & img, double* confidence)
    {
        return track(img, lastPosition.x, lastPosition.y, lastPosition.x + lastPosition.width, lastPosition.y + lastPosition.height, confidence);
    }
    void update();
};

#endif // COLOTRACKER__H
