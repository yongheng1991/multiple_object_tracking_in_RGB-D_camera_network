#ifndef Object_Detector_H
#define Object_Detector_H
//#include <QMessageBox>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include <opt_msgs/Track3DArray.h>
#include <mutex>
#include <thread>
using namespace cv;
class Object_Detector
{
public:



    //    ///////////For hsv mask///////////
    static int HMin, SMin, VMin,DMin;
    static int HMax,SMax, VMax,DMax;
    //    ///////////For hsv mask///////////


   //////////////////////////For camshift//////////////////////////
   static std::string Backprojection_Mode;
   //    Backprojection_Mode{
   //        H ,//just use hue of select roi to do calhist for one time, then use the permenet hue hist to do the backproject on every hue frame
   //        HS,//use hue and saturation of select roi to do calhist(named:HS_hist) for one time, then use the permenet hs hist to do the backproject on every hue&saturation frame
   //        HSD//use hue and saturation of select roi to do calhist for one time(got :hs_hist_for_HSD,it is just used under occlusion), calculate the pdf of this hist(got hs_hist_pdf),
   //        //then use the roi of depth data to do calhist and pdf(got :tmp_depth_hist_pdf) for every frame,
   //        //then combine the permenet hs hist pdf (named:hs_hist_pdf) and tmp depth hist(named: tmp_depth_hist_pdf) to do the backproject on every hsd frame,when occluded
   //    };

    static int h_bins,s_bins,v_bins;//for the hue backrojection hist ,divide (HMin,HMax) to h_bins parts
    static int d_bins;// depth bins,
    //////////////////////////For camshift//////////////////////////

    double depth_mu,depth_mu_value;
    double depth_sigma;
    int occ_frames;


    ///////////For camshift recover from occlusion///////////
    //for camshift, for the window in backproject image,the value represent the similirity with the ROI. the value of the pixel is equal to 255*percentage ,the percentage come from the histogram(problility)
    // QUALITY_TOLERANCE here means the TOLERANCE of sum of the window in backproject image,it represent the quality of the window.
    // DENSITY_TOLORENCE here means the TOLERANCE of the DENSITY,DENSITY=QUALITY/window.area();
    // AREA_TOLERANCE here means the TOLERANCE of the x,y ,window.it means the objects maybe apprear in the next frame ,in the window which increase it's size by AREA_TOLERANCE every frame.so the bigger AREA_TOLERANCE,the objects can move faster.
    // if the QUALITY<QUALITY_TOLERANCE, the object is "totally occluded",
    // then when QUALITY<QUALITY_TOLERANCE, it will appeare, but at first ,not totaly appear,the window may be much bigger then the size of object,
    // then we calculate the DENSITY,if DENSITY<DENSITY_TOLORENCE.it means the window not just included the object,and the objects are "half occluded"
    // then the camshift will decrese the size of the window,untill DENSITY>DENSITY_TOLORENCE,it will be "not occluded";
    static int AREA_TOLERANCE;//AREA_TOLERANCE is also used to create the position_maks
    static int QUALITY_TOLERANCE;
    static double DENSITY_TOLORENCE;
    static bool with_occ_handle;

    bool occluded;//if the QUALITY<QUALITY_TOLERANCE
    bool half_occluded;//if QUALITY>QUALITY_TOLERANCE , DENSITY<DENSITY_TOLORENCE
    double half_occluded_frames;//once the target is re-localized, it may keep half_occluded because of the similar background, when half_occluded_frames>10, we will assume the tracker is difted

    bool recover_from_occ;//(occ->_not_occ or half_occ->not_occ)
    double occluded_frames;//number of frames begain from the first occlusion(not_occ->occ or not_occ->half_occ),  and last occlusion(occ->_not_occ or half_occ->not_occ)
    cv::Rect window_before_occ;//tracked window before occlusion

    bool drifted; //when recover_from_occ=true, calculate the 2d distance between the "window_before_occ" and "recovered window", if distance>occluded_frames*30, we assume the tracker drift to another similar object which is far away from the target
    //if drifted=true, update the window with the tracker msg
    ///////////For camshift recover from occlusion///////////





    static std::vector<Rect> current_detected_boxes;//this one come from class:multiple_objects_detection, every time of every object's detection ,multiple_objects_detection will update this varable ,we use it to generate the other_object_mask
    cv::Mat roi_from_file;// the roi come from file which is set when "select_rois_from_file"
    std::string object_name;
    cv::Rect last_tracked_window_without_occ;


    static std::vector<std::pair<std::string,cv::Point2d>>current_track2D_array;
private:
//    std::mutex lock;
    static cv::Mat mainColor;
    static cv::Mat mainDepth;
    cv::Mat Color,Depth;//copied from mainColor,mainDepth



    bool firstRun;
    cv::Rect currentRect;// just used in the  initilization(no important)
    cv::Rect selection;// just used in the  initilization(no important)

    cv::RotatedRect current_detectedBox;//detction result (important)
    cv::Rect detectWindow;// used as search window at the beginning of detection(important)


    //////////main variable for calculate the histogram ///////////
    cv::Mat hsv, hue, hsv_mask,backproj;
    cv::Mat h_hist;//H mode
    cv::Mat hs_hist_for_HS;//HS mode
    cv::Mat hs_hist_for_HSD,//just used under occlusion,because ,when oclluded ,can't use depth infomation to detect the objects
    hs_hist_pdf,//use hs_hist_for_HSD to get pdf ,use it as the permnet part
    tmp_depth_hist_pdf,//use every depth frame to get this ,use it as the tmp part
    hsd_hist,//combine hs_hist_pdf(perment) and  tmp_depth_hist_pdf(tmp,every frame) to get this
    tmp_value_hist_pdf,
    hsvd_hist;
    cv::Mat hsv_hist,hs_hist_for_HSV;
    Mat s_hsv_hist;


    Mat other_object_mask,position_mask,depth_mask;//some mask that used for get the good back projection image



public:
    Object_Detector()
        :firstRun(true),occluded(false),half_occluded(false),half_occluded_frames(10),depth_mu(255),depth_sigma(5),occ_frames(0),drifted(false),recover_from_occ(false),occluded_frames(0)
    {}

    static void setMainColor(const cv::Mat _mainColor);
    static cv::Mat getMainColor();

    static void setMainDepth(const cv::Mat _mainDepth);
    static cv::Mat getMainDepth();

    void setCurrentRect(const cv::Rect _currentRect);
    cv::Rect getCurrentRect();

    void setObjectName(const std::string object_name);
    std::string getObjectName();

    void setcurrent_detected_boxes(std::vector<Rect> _current_detected_boxes);
    std::vector<Rect>  getcurrent_detected_boxes();

    static void setCurrentTrackArray(const std::vector<std::pair<std::string,cv::Point2d>>_current_track2D_array);
    static std::vector<std::pair<std::string,cv::Point2d>>getCurrentTrackArray();

    void H_backprojection();
    void HS_backprojection();
    void HSD_backprojection();
    void HSV_backprojection();
    void HSVD_backprojection(int id);
    void RGBD_backprojection();
    cv::RotatedRect object_shift(InputArray _probColor,Rect& window, TermCriteria criteria);//camshift + occlusion handle
    cv::RotatedRect detectCurrentRect(int id);//main detection function
};

//stastic varable defination
int Object_Detector::HMax;
int Object_Detector::SMax;
int Object_Detector::VMax;
int Object_Detector::HMin;
int Object_Detector::SMin;
int Object_Detector::VMin;

int Object_Detector::DMax;
int Object_Detector::DMin;

int Object_Detector::h_bins;
int Object_Detector::s_bins;
int Object_Detector::v_bins;
int Object_Detector::d_bins;

int Object_Detector::AREA_TOLERANCE;
int Object_Detector::QUALITY_TOLERANCE;
double Object_Detector::DENSITY_TOLORENCE;

std::vector<Rect> Object_Detector::current_detected_boxes;
std::vector<std::pair<std::string,cv::Point2d>>Object_Detector::current_track2D_array;

std::string Object_Detector::Backprojection_Mode;
bool Object_Detector::with_occ_handle;

#endif // Object_Detector_H
