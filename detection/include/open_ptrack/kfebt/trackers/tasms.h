#ifndef TASMS_H
#define TASMS_H

#include "btracker.h"
#include "ASMS/colotracker_.h"
//#include"open_ptrack/asms/colotracker.h"

class tASMS : public BTracker
{
public:
    tASMS(float dist_adj = DIST_ADJ, float conf_adj = 1.0);

    void init(cv::Mat& image, cv::Rect region);
    void correctState(std::vector<float> st);
    void track();
    void update();
    void newFrame(cv::Mat& image, std::vector<float> predictRect);
    cv::Rect getRect();

    virtual void whoIam(){
        std::cout << "I am tASMS" << std::endl;
    }

private:
    ColorTrackert asms;
    cv::Mat currentFrame;
    std::vector<float> currentPredictRect;
};

#endif // TASMS_H
