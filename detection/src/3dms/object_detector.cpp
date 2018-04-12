#include "open_ptrack/3dms/object_detector.h"
#include <iostream>
// Static member definition ...
cv::Mat Object_Detector::mainColor;
cv::Mat Object_Detector::mainDepth;

void Object_Detector::setMainColor(const cv::Mat _mainColor)
{
    _mainColor.copyTo(mainColor);
}

cv::Mat Object_Detector::getMainColor()
{
    return mainColor;
}

void Object_Detector::setMainDepth(const cv::Mat _mainDepth)
{
    _mainDepth.copyTo(mainDepth);
}

cv::Mat Object_Detector::getMainDepth()
{
    return mainDepth;
}


void Object_Detector::setCurrentRect(const cv::Rect _currentRect)
{
    currentRect = _currentRect;
    selection = currentRect;
}

cv::Rect Object_Detector::getCurrentRect()
{
    return currentRect;
}


void Object_Detector::setObjectName(const std::string _object_name)
{
    object_name=_object_name;
}
std::string Object_Detector::getObjectName()
{
    return object_name;
}


void Object_Detector::setcurrent_detected_boxes(std::vector<Rect> _current_detected_boxes)
{
    current_detected_boxes=_current_detected_boxes;
}

std::vector<Rect> Object_Detector::getcurrent_detected_boxes()
{
    return current_detected_boxes;
}

void Object_Detector::setCurrentTrackArray(std::vector<std::pair<std::string,cv::Point2d>>msgs)
{
    current_track2D_array=msgs;
}

std::vector<std::pair<std::string,cv::Point2d>>Object_Detector::getCurrentTrackArray()
{
    return current_track2D_array;
}


void non_maxima_suppression(const Mat& src, Mat& mask, const bool remove_plateaus){

    //find pixels that are equal to the local neighborhood not maximum (including plateaus)
    dilate(src, mask, Mat());
    compare(src, mask, mask, CMP_GE);

    //optionally filter out pixels that are equal to the local minimum ('plateaus')
    if(remove_plateaus){
        Mat non_plateau_mask;
        erode(src, non_plateau_mask, Mat());
        compare(src, non_plateau_mask, non_plateau_mask, CMP_GT);
        bitwise_and(mask, non_plateau_mask, mask);
    }
}
//function that finds the peaks of a given hist image
void findHistPeaks(Mat _src, OutputArray _idx, const float scale= 0.05, const Size& ksize = Size(9,9), const bool remove_plateus = true){

    Mat hist = _src.clone();
    //  cout<<"hist_in peaks0: "<<hist<<endl<<endl;

    // find the min and max values of the hist image
    double min_val, max_val;
    minMaxLoc(hist, &min_val, &max_val);
    //  cout<<"hist_in peaks1: "<<hist<<endl<<endl;

    Mat mask;
    //  GaussianBlur(hist, hist, ksize, 0); //smooth a bit in otder to obtain better result
    //  cout<<"hist_in peaks2: "<<hist<<endl<<endl;

    non_maxima_suppression(hist, mask, remove_plateus);

    //  cout<<"mask_in peaks2: "<<mask<<endl<<endl;


    //    imshow("mask",mask);
    //    cv::waitKey(10);
    std::vector<Point> maxima; // Outputs, locations of non-zero pixels

    int count = countNonZero(mask);
    if(count > 0)
    {
        cv::findNonZero(mask, maxima);
    }
    else{
        return;
    }

    for(std::vector<Point>::iterator it = maxima.begin(); it != maxima.end();){
        Point pnt = *it;
        float pVal = hist.at<float>(/*pnt.x,*/pnt.y -1);
        float val = hist.at<float>(/*pnt.x, */ pnt.y);
        float nVal = hist.at<float>(/*pnt.x, */pnt.y+1);

        // filter peaks
        if((val > max_val * scale))
        {
            it->x=val;
            ++it;

        }
        else
            it = maxima.erase(it);
    }

    Mat(maxima).copyTo(_idx);

}

void Object_Detector::H_backprojection()
{
    float h_ranges[] = {0,(float)HMax};
    const float* ph_ranges = h_ranges;
    int h_channels[] = {0, 0};
    hue.create(Color.size(), Color.depth());
    cv::mixChannels(&Color, 1, &hue, 1, h_channels, 1);

    if( firstRun )
    {
        if(!occluded)
        {
            cv::Mat roi(hue, selection), maskroi(hsv_mask, selection);
            cv::calcHist(&roi, 1, 0, maskroi, h_hist, 1, &h_bins, &ph_ranges);
            cv::normalize(h_hist, h_hist, 0, 255, CV_MINMAX);
            detectWindow = selection;
            firstRun = false;
            //        std::cout<<"H mode"<<std::endl;
        }
        else{
            Mat roi_color= roi_from_file;
            //            cv::imshow("roi from file",roi_color);
            //            cv::waitKey(30);
            Mat roi_hsv,roi_hsv_mask;
            cv::cvtColor(roi_color, roi_hsv, CV_BGR2HSV);
            cv::inRange(roi_hsv, cv::Scalar(HMin, SMin, MIN(VMin,VMax)),
                        cv::Scalar(HMax, SMax, MAX(VMin, VMax)), roi_hsv_mask);
            cv::calcHist(&roi_hsv, 1, h_channels, roi_hsv_mask, h_hist, 1, &h_bins, &ph_ranges);
            cv::normalize(h_hist, h_hist, 0, 255, CV_MINMAX);
            detectWindow = selection;
            firstRun = false;
        }

    }
    cv::calcBackProject(&hue, 1, 0, h_hist, backproj, &ph_ranges,1,true);
}

void Object_Detector::HS_backprojection()
{
    int hs_size[] = { h_bins, s_bins };
    float h_range[] = {(float)HMin,(float)HMax};
    float s_range[] = { (float)SMin, (float)SMax };
    const float* phs_ranges[] = { h_range, s_range };
    int hs_channels[] = { 0, 1 };

    if( firstRun )
    {
        if(!occluded)
        {
            cv::Mat roi(Color, selection), maskroi(hsv_mask, selection);
            cv::calcHist(&roi, 1, hs_channels, maskroi, hs_hist_for_HS, 2, hs_size, phs_ranges, true, false);
            cv::normalize(hs_hist_for_HS, hs_hist_for_HS, 0, 255, CV_MINMAX);
            detectWindow = selection;
            firstRun = false;
            //std::cout<<"HS mode"<<std::endl;
        }
        else{
            Mat roi_color= roi_from_file;
            //            cv::imshow("roi from file",roi_color);
            //            cv::waitKey(30);
            Mat roi_hsv,roi_hsv_mask;
            cv::cvtColor(roi_color, roi_hsv, CV_BGR2HSV);
            cv::inRange(roi_hsv, cv::Scalar(HMin, SMin, MIN(VMin,VMax)),
                        cv::Scalar(HMax, SMax, MAX(VMin, VMax)), roi_hsv_mask);
            cv::calcHist(&roi_hsv, 1, hs_channels, roi_hsv_mask, hs_hist_for_HS, 2, hs_size, phs_ranges, true, false);
            cv::normalize(hs_hist_for_HS , hs_hist_for_HS , 0, 255, CV_MINMAX);
            detectWindow = selection;
            firstRun = false;
        }
    }
    cv::calcBackProject( &Color, 1, hs_channels, hs_hist_for_HS, backproj, phs_ranges, 1, true );

}

void Object_Detector::HSV_backprojection()
{
    int v_bins=18;
    const int hsv_size[] = { h_bins, s_bins ,v_bins};
    const int hs_size[] =  { h_bins, s_bins };

    float h_range[] = { (float)HMin, (float)HMax };
    float s_range[] = { (float)SMin, (float)SMax };
    float v_range[] = { (float)VMin, (float)VMax };

    const float* pv_ranges= v_range;
    const float* phs_ranges[] = { h_range, s_range };
    const float* phsv_ranges[] = {h_range, s_range ,v_range };

    int hs_channels[] = { 0, 1 };
    int hsv_channels[] = {0,1,2};
    int v_channels[] = {2};
    if( firstRun )
    {
        cv::Mat roi(Color, selection), maskroi(hsv_mask, selection);
        cv::calcHist(&roi, 1, hs_channels, maskroi, hs_hist_for_HSV, 2, hs_size, phs_ranges, true, false);
        cv::normalize(hs_hist_for_HSV , hs_hist_for_HSV , 0, 255, CV_MINMAX);

        double sum_hs_hist=sum(hs_hist_for_HSV)[0];
        hs_hist_pdf=hs_hist_for_HSV/sum_hs_hist;

        //used to generate the initial hsd_hist(use this just for the right data format )
        cv::calcHist(&roi, 1, hsv_channels, maskroi, hsv_hist, 3, hsv_size, phsv_ranges, true, false);



        cv::calcHist(&roi, 1, v_channels, maskroi, tmp_depth_hist_pdf, 1, &v_bins, &pv_ranges);
        double sum_tmp_depth_hist_pdf=sum(tmp_depth_hist_pdf)[0];
        tmp_depth_hist_pdf=tmp_depth_hist_pdf/sum_tmp_depth_hist_pdf;

        for (int i=0; i<hsv_size[0]; i++) {
            for (int j=0; j<hsv_size[1]; j++) {
                for (int k=0; k<hsv_size[2]; k++) {
                    hsv_hist.at<float>(i,j,k)=255*hs_hist_pdf.at<float>(i,j)*tmp_depth_hist_pdf.at<float>(k);
                }
            }
        }
        //normalize hsd_hist
        double hMin,hMax;
        minMaxIdx(hsv_hist, &hMin, &hMax);
        hsv_hist = 255 * hsv_hist / hMax;
        //        std::cout<<hsd_hist.channels()<<std::endl;
        //        std::cout<<hMax<<std::endl;



        //        hsd_hist_tmp.clear();
        //        Mat hsd_hist_tmp_;
        //        for (int k=0; k<hsd_size[2]; k++) {
        //            hsd_hist_tmp_=tmp_depth_hist_pdf.at<float>(k)*255*hs_hist_pdf;
        //            hsd_hist_tmp.push_back(hsd_hist_tmp_);
        //        }
        //        Mat hsd_hist_mat;
        //        merge(hsd_hist_tmp,hsd_hist_mat);
        ////        double hMin,hMax;
        //        minMaxIdx(hsd_hist_mat, &hMin, &hMax);
        //       hsd_hist_mat = 255 * hsd_hist_mat / hMax;



        //        SparseMat hsd_hist_sp(3, hsd_size, hs_hist_pdf.type());

        detectWindow = selection;
        firstRun = false;
    }
    /*
    else
    {
        if(!occluded&&!half_occluded)
        {

            cv::Mat roi(hsv, detectWindow), maskroi(hsv_mask, detectWindow);
            cv::calcHist(&roi, 1, v_channels, maskroi, tmp_depth_hist_pdf, 1, &v_bins, &pv_ranges);
            double sum_tmp_depth_hist_pdf=sum(tmp_depth_hist_pdf)[0];
            tmp_depth_hist_pdf=tmp_depth_hist_pdf/sum_tmp_depth_hist_pdf;
            for (int i=0; i<hsv_size[0]; i++) {
                for (int j=0; j<hsv_size[1]; j++) {
                    for (int k=0; k<hsv_size[2]; k++) {
                        hsv_hist.at<float>(i,j,k)=255*hs_hist_pdf.at<float>(i,j)*tmp_depth_hist_pdf.at<float>(k);
                    }
                }
            }
            //normalize hsd_hist
            double hMin,hMax;
            minMaxIdx(hsv_hist, &hMin, &hMax);
            hsv_hist = 255 * hsv_hist / hMax;
            std::cout<<"HSV hist update"<<std::endl;
        }

    }

*/
    cv::calcBackProject( &Color, 1, hsv_channels, hsv_hist, backproj, phsv_ranges, 1, true );


}

void Object_Detector::HSD_backprojection()
{

    const int hsd_size[] = { h_bins, s_bins ,d_bins};
    const int hs_size[] =  { h_bins, s_bins };

    float h_range[] = { (float)HMin, (float)HMax };
    float s_range[] = { (float)SMin, (float)SMax };
    float d_range[] = { 0, 255 };

    const float* pd_ranges= d_range;
    const float* phs_ranges[] = { h_range, s_range };
    const float* phsd_ranges[] = {h_range, s_range ,d_range };

    int hs_channels[] = { 0, 1 };
    int hsd_channels[] = {0,1,2};
    int hsd_channels_formix[] = {0, 2};

    cv::mixChannels(&Depth, 1, &Color, 3, hsd_channels_formix, 1);//hsv-->hsd

    if( firstRun )
    {
        if(!occluded)//if the roi is from the file, the first frame will be set to occluded, in this situation,we just calculate the hs pdf and use it to search the object
        {
            cv::Mat roi(Color, selection), maskroi(hsv_mask, selection);
            cv::calcHist(&roi, 1, hs_channels, maskroi, hs_hist_for_HSD, 2, hs_size, phs_ranges, true, false);
            cv::normalize(hs_hist_for_HSD , hs_hist_for_HSD , 0, 255, CV_MINMAX);

            double sum_hs_hist=sum(hs_hist_for_HSD)[0];
            hs_hist_pdf=hs_hist_for_HSD/sum_hs_hist;

            //used to generate the initial hsd_hist(use this just for the right data format )
            cv::calcHist(&roi, 1, hsd_channels, maskroi, hsd_hist, 3, hsd_size, phsd_ranges, true, false);

            cv::Mat roi_depth(Depth, selection);

            //calculate the the current_trackBox(rotatedrect) mask,named depth_mask(in this mask ,just the the value in the area :current_detectBox(rotatedrect) is 255)
            Point2f vertices[4];
            current_detectedBox.points(vertices);
            std::vector< std::vector<Point> >  co_ordinates;
            co_ordinates.push_back(std::vector<Point>());
            co_ordinates[0].push_back(vertices[0]);
            co_ordinates[0].push_back(vertices[1]);
            co_ordinates[0].push_back(vertices[2]);
            co_ordinates[0].push_back(vertices[3]);
            depth_mask=Mat::zeros(Color.size(),CV_8UC1);
            drawContours( depth_mask,co_ordinates,0, Scalar(255),CV_FILLED, 8 );
            depth_mask&=hsv_mask;
            cv::Mat  depth_mask_roi(depth_mask, detectWindow);
            cv::calcHist(&roi_depth, 1, 0, depth_mask_roi, tmp_depth_hist_pdf, 1, &d_bins, &pd_ranges);
            double sum_tmp_depth_hist_pdf=sum(tmp_depth_hist_pdf)[0];
            tmp_depth_hist_pdf=tmp_depth_hist_pdf/sum_tmp_depth_hist_pdf;


            //combine the hs and depth
            for (int i=0; i<hsd_size[0]; i++) {
                for (int j=0; j<hsd_size[1]; j++) {
                    for (int k=0; k<hsd_size[2]; k++) {
                        hsd_hist.at<float>(i,j,k)=255*hs_hist_pdf.at<float>(i,j)*tmp_depth_hist_pdf.at<float>(k);
                    }
                }
            }

            //normalize hsd_hist
            double hMin,hMax;
            minMaxIdx(hsd_hist, &hMin, &hMax);
            hsd_hist = 255 * hsd_hist / hMax;

            detectWindow = selection;
            firstRun = false;
        }
        else{ // if we get the roi from file ,we just calculate the hs pdf

            Mat roi_color= roi_from_file;
            Mat roi_hsv,roi_hsv_mask;
            cv::cvtColor(roi_color, roi_hsv, CV_BGR2HSV);
            cv::inRange(roi_hsv, cv::Scalar(HMin, SMin, MIN(VMin,VMax)),
                        cv::Scalar(HMax, SMax, MAX(VMin, VMax)), roi_hsv_mask);
            //            imshow("roi_hsv_mask",roi_hsv_mask);
            //            cv::waitKey(10);

            cv::calcHist(&roi_hsv, 1, hs_channels, roi_hsv_mask, hs_hist_for_HSD, 2, hs_size, phs_ranges, true, false);
            cv::normalize(hs_hist_for_HSD , hs_hist_for_HSD , 0, 255, CV_MINMAX);

            double sum_hs_hist=sum(hs_hist_for_HSD)[0];
            hs_hist_pdf=hs_hist_for_HSD/sum_hs_hist;

            //used to generate the initial hsd_hist(use this just for the right data format )
            cv::calcHist(&roi_hsv, 1, hsd_channels, roi_hsv_mask, hsd_hist, 3, hsd_size, phsd_ranges, true, false);

            detectWindow = selection;
            firstRun = false;
        }
    }

    else//main loop to get the hsd pdf, just update the depth pdf , and combine it with the initial hs pdf
    {
        if(!occluded&&!half_occluded)
        {
            cv::Mat roi_depth(Depth, detectWindow);

            //calculate the the current_trackBox(rotatedrect) mask,named depth_mask(in this mask ,just the the value in the area :current_detectBox(rotatedrect) is 255)
            Point2f vertices[4];
            current_detectedBox.points(vertices);
            std::vector< std::vector<Point> >  co_ordinates;
            co_ordinates.push_back(std::vector<Point>());
            co_ordinates[0].push_back(vertices[0]);
            co_ordinates[0].push_back(vertices[1]);
            co_ordinates[0].push_back(vertices[2]);
            co_ordinates[0].push_back(vertices[3]);

            depth_mask=Mat::zeros(Color.size(),CV_8UC1);
            drawContours( depth_mask,co_ordinates,0, Scalar(255),CV_FILLED, 8 );
            depth_mask&=hsv_mask;

            cv::Mat  depth_mask_roi(depth_mask, detectWindow);

            cv::calcHist(&roi_depth, 1, 0, depth_mask_roi, tmp_depth_hist_pdf, 1, &d_bins, &pd_ranges);
            double sum_tmp_depth_hist_pdf=sum(tmp_depth_hist_pdf)[0];
            tmp_depth_hist_pdf=tmp_depth_hist_pdf/sum_tmp_depth_hist_pdf;

            //////////////////////cost a lot of time  //////////////////////
            for (int i=0; i<hsd_size[0]; i++) {
                for (int j=0; j<hsd_size[1]; j++) {
                    for (int k=0; k<hsd_size[2]; k++) {
                        hsd_hist.at<float>(i,j,k)=255*hs_hist_pdf.at<float>(i,j)*tmp_depth_hist_pdf.at<float>(k);
                    }
                }
            }
            //////////////////////rcost a lot of time  //////////////////////

            double hMin,hMax;
            minMaxIdx(hsd_hist, &hMin, &hMax);
            hsd_hist = 255 * hsd_hist / hMax;
        }
    }


    if(!occluded&&!half_occluded)//if not occluded, use hsd pdf
    {
        cv::calcBackProject( &Color, 1, hsd_channels, hsd_hist, backproj, phsd_ranges, 1, true );
    }
    else//if occluded, use hs pdf
    {
        cv::calcBackProject( &Color, 1, hs_channels, hs_hist_for_HSD, backproj, phs_ranges, 1, true );
    }

}

void Object_Detector::HSVD_backprojection(int id)
{
    const int hsv_size[] = { h_bins, s_bins ,v_bins};
    const int hs_size[] =  { h_bins, s_bins };

    float h_range[] = { (float)HMin, (float)HMax };
    float s_range[] = { (float)SMin, (float)SMax };//the upper boundary is exclusive
    float v_range[] = { (float)VMin, (float)VMax};
    float d_range[] = { (float)DMin, (float)DMax};

    const float* pd_ranges= d_range;
    const float* pv_ranges= v_range;
    const float* phs_ranges[] = { h_range, s_range };
    const float* phsv_ranges[] = {h_range, s_range ,v_range };

    int hs_channels[] = { 0, 1 };
    int hsv_channels[] = {0,1,2};
    int v_channels[] = {2};

    if( firstRun )
    {
        //        depth_mu=255;
        //        depth_sigma=5;

        //        Mat color_rgb;
        //        cvtColor(Color,color_rgb,CV_HSV2BGR);

        //        cv::rectangle(color_rgb, selection, cv::Scalar(250*(1), 250*(1-1), 250*(1-2)), 2, CV_AA);
        //        cv::imshow( "show_search_window", Color );
        //        waitKey(2);

        cv::Mat roi_depth(Depth, selection);
        //        cv::Mat depth_mask_=Mat::ones(Color.size(),CV_8UC1);
        //        cv::Mat  depth_mask_roi(depth_mask_, selection);
        cv::calcHist(&roi_depth, 1, 0, cv::Mat(),tmp_depth_hist_pdf, 1, &d_bins, &pd_ranges);
        tmp_depth_hist_pdf.at<float>(0,0)=0;
        tmp_depth_hist_pdf.at<float>(0,255)=0;

        std::vector<Point> peaks;
        findHistPeaks(tmp_depth_hist_pdf, peaks);
        for(std::vector<Point>::iterator it= peaks.begin();it!=peaks.end();it++)
        {
            if(depth_mu>(*it).y&&(*it).y>5)
            {
                depth_mu=(*it).y;
                depth_mu_value=(*it).x;
            }
        }

        Mat tmp_depth_hist_pdf_=Mat::zeros(tmp_depth_hist_pdf.size(),tmp_depth_hist_pdf.type());
        //        int it_start=(depth_mu>depth_sigma)?(depth_mu-depth_sigma):0;

        for(int i =0;i<depth_mu+3;i++)
        {
            tmp_depth_hist_pdf_.at<float>(i,1)=tmp_depth_hist_pdf.at<float>(i,1);
        }
        double sum_tmp_depth_hist_pdf=sum(tmp_depth_hist_pdf_)[0];
        tmp_depth_hist_pdf=tmp_depth_hist_pdf_/sum_tmp_depth_hist_pdf;

        cv::Mat roi(Color, selection);
        Mat depth_roi=Depth(selection),depth_roi_mask;
        //       imshow("depth_roi",depth_roi);

        //        waitKey(100000);
        int _depth_sigma=0;

        for(; _depth_sigma<10;_depth_sigma++)
        {

            cv::inRange(depth_roi, cv::Scalar(0),cv::Scalar(depth_mu+_depth_sigma), depth_roi_mask);
            int count_white = cv::countNonZero(depth_roi_mask == 255);
            //std::cout<<"(float(count_white))/(selection.area())"<<(float(count_white))/(selection.area())<<std::endl;
            if((float(count_white))/(selection.area())>0.6)
                break;
        }


        //        cv::inRange(Depth, cv::Scalar(0),cv::Scalar(depth_mu+_depth_sigma), depth_mask);
        //        imshow("depth_mask",depth_mask);


        Mat maskroi= depth_roi_mask;
        Mat check_roi;
        roi.copyTo(check_roi,maskroi);
        //imshow("check_roi",check_roi);

        //waitKey(1);
        ///
        ///
        //use one line to calculate the hsv_hist
        cv::calcHist(&roi, 1, hsv_channels, maskroi, hsv_hist, 3, hsv_size, phsv_ranges, true, false);
        //        cv::calcHist(&roi, 1, hsv_channels, Mat(), hsv_hist, 3, hsv_size, phsv_ranges, true, false);

        //use loop to calculate the hsv)hist
        //                cv::calcHist(&roi, 1, hs_channels, maskroi, hs_hist_for_HSV, 2, hs_size, phs_ranges, true, false);
        //                cv::normalize(hs_hist_for_HSV , hs_hist_for_HSV , 0, 255, CV_MINMAX);

        //                double sum_hs_hist=sum(hs_hist_for_HSV)[0];
        //                hs_hist_pdf=hs_hist_for_HSV/sum_hs_hist;

        //                cv::calcHist(&roi, 1, hsv_channels, maskroi, hsv_hist, 3, hsv_size, phsv_ranges, true, false);

        //                cv::calcHist(&roi, 1, v_channels, maskroi, tmp_value_hist_pdf, 1, &v_bins, &pv_ranges);
        //                double sum_tmp_value_hist_pdf=sum(tmp_value_hist_pdf)[0];
        //                tmp_value_hist_pdf=tmp_value_hist_pdf/sum_tmp_value_hist_pdf;

        //                for (int i=0; i<hsv_size[0]; i++) {
        //                    for (int j=0; j<hsv_size[1]; j++) {
        //                        for (int k=0; k<hsv_size[2]; k++) {
        //                            hsv_hist.at<float>(i,j,k)=255*hs_hist_pdf.at<float>(i,j)*tmp_value_hist_pdf.at<float>(k);
        //                        }
        //                    }
        //                }
        //        hsv_hist=hsv_hist/sum(hsv_hist)[0];



        //calculate the surrounding hist

        ///  ///  ///  ///use the surrounding/background to weight the color model
        ///
        cv::Point inflationPoint(-(selection.width/2), -(selection.height/2));
        cv::Size inflationSize(selection.width, selection.height);
        //        cv::Point inflationPoint(-(selection.width), -(selection.height));
        //        cv::Size inflationSize(selection.width*2, selection.height*2);
        Rect surrounding_rect = selection+inflationPoint;
        surrounding_rect += inflationSize;
        surrounding_rect=surrounding_rect&Rect(0,0,Color.size().width,Color.size().height);

        ////think twice about this , try not to weaker the target model,
        /// combine two bg model???????????????????????????????
        cv::Mat s_roi(Color, surrounding_rect);

        //                cv::inRange(Depth, cv::Scalar(0),cv::Scalar(depth_mu+_depth_sigma), depth_mask);
        //                Mat inv_depth_mask=255-depth_mask;

//        Mat inv_depth_mask=Mat::zeros(Color.size(),CV_8UC1);
//        Mat _inv_depth_mask=255*Mat::ones(Color.size(),CV_8UC1);
//        inv_depth_mask(selection).copyTo(_inv_depth_mask(selection));

         Mat inv_depth_mask=255*Mat::ones(Color.size(),CV_8UC1);
         inv_depth_mask(selection)=(uchar)0;


        //add other object mask here for the surroundings in case there are two same object stay close to each other
        //        other_object_mask=Mat::ones(Color.size(), CV_8U)*255;
        //        for (int i=0; i<current_detected_boxes.size(); i++)
        //        {
        //            if(i!=id)
        //            {
        //                uchar tmp =0;
        //                Rect current_tracked_box =current_detected_boxes[i];
        //                current_tracked_box=current_tracked_box&Rect(0,0,Color.size().width,Color.size().height);
        //                other_object_mask(current_tracked_box)=tmp;
        //            }
        //        }
        //        inv_depth_mask=inv_depth_mask&other_object_mask;


        //  imshow("_inv_depth_mask",inv_depth_mask);
        cv::Mat s_maskroi(inv_depth_mask, surrounding_rect);

        //use one line to calculate the s-hsv_hist
        cv::calcHist(&s_roi, 1, hsv_channels, s_maskroi, s_hsv_hist, 3, hsv_size, phsv_ranges, true, false);


        //                //use loop to calculate the hs*v hist
        //                Mat _tmp_hs_hist;
        //                cv::calcHist(&s_roi, 1, hs_channels, s_maskroi, _tmp_hs_hist, 2, hs_size, phs_ranges, true, false);
        //                cv::normalize(_tmp_hs_hist , _tmp_hs_hist , 0, 255, CV_MINMAX);
        //                double _sum_hs_hist=sum(_tmp_hs_hist)[0];
        //                Mat _tmp_hs_hist_pdf;
        //                _tmp_hs_hist_pdf=_tmp_hs_hist/_sum_hs_hist;
        //                Mat _tmp_value_hist;
        //                cv::calcHist(&s_roi, 1, v_channels, s_maskroi, _tmp_value_hist, 1, &v_bins, &pv_ranges);
        //                double _sum_tmp_value_hist=sum(_tmp_value_hist)[0];
        //                Mat _tmp_value_hist_pdf;
        //                _tmp_value_hist_pdf=_tmp_value_hist/_sum_tmp_value_hist;
        //                for (int i=0; i<hsv_size[0]; i++) {
        //                    for (int j=0; j<hsv_size[1]; j++) {
        //                        for (int k=0; k<hsv_size[2]; k++) {
        //                            s_hsv_hist.at<float>(i,j,k)=255*_tmp_hs_hist_pdf.at<float>(i,j)*_tmp_value_hist_pdf.at<float>(k);
        //                        }
        //                    }
        //                }

        //        s_hsv_hist=s_hsv_hist/sum(s_hsv_hist)[0];//no need to do this,


        double s_hsv_hist_min,s_hsv_hist_max;
        Mat s_hsv_hist_mask_nonzero = s_hsv_hist>0;
        cv::minMaxIdx(s_hsv_hist, &s_hsv_hist_min, &s_hsv_hist_max,NULL,NULL,s_hsv_hist_mask_nonzero);
        //std::cout<<"s_hsv_hist_min: "<<s_hsv_hist_min<<std::endl;
        //        double sum = 0;
        //        for (int i=0; i<hsv_size[0]; i++) {
        //            for (int j=0; j<hsv_size[1]; j++) {
        //                for (int k=0; k<hsv_size[2]; k++) {

        //                    double background_weight;
        //                    if(s_hsv_hist.at<float>(i,j,k)>0)
        //                        background_weight=(s_hsv_hist_min)/(s_hsv_hist.at<float>(i,j,k));
        //                    else
        //                        background_weight=1;

        //                    hsv_hist.at<float>(i,j,k)=(hsv_hist.at<float>(i,j,k))*background_weight;

        //                }
        //            }
        //        }

        /////the loop can be equal to
        ///
        s_hsv_hist.setTo(s_hsv_hist_min,s_hsv_hist==0);
        hsv_hist=hsv_hist*(s_hsv_hist_min)/(s_hsv_hist);
        //        hsv_hist=hsv_hist*(s_hsv_hist_min+1)/(s_hsv_hist+1);


        double hMin,hMax;
        minMaxIdx(hsv_hist, &hMin, &hMax);
        hsv_hist = 255 * hsv_hist / hMax;


        //        Mat hsv_hist_mask_low=hsv_hist>5;
        //        Mat _hsv_hist;
        //        hsv_hist.copyTo(_hsv_hist,hsv_hist_mask_low);
        //        hsv_hist=_hsv_hist;


        cv::calcBackProject( &Color, 1, hsv_channels, hsv_hist, backproj, phsv_ranges, 1, true );
        //        imshow("hsv_back",backproj);
        Mat backproj_tmp_depth;
        cv::normalize(tmp_depth_hist_pdf, tmp_depth_hist_pdf, 0, 255, CV_MINMAX);
        cv::calcBackProject( &Depth, 1, 0, tmp_depth_hist_pdf, backproj_tmp_depth, &pd_ranges, 1, true );
        //        imshow("backproj_tmp_depth",backproj_tmp_depth);
        backproj.convertTo(backproj,CV_16U);
        backproj_tmp_depth.convertTo(backproj_tmp_depth,CV_16U);
        backproj=backproj.mul(backproj_tmp_depth);
        backproj.convertTo(backproj, CV_8U,255.0/65536.0,0);

        detectWindow = selection;
        firstRun = false;

    }

    else
    {
        if(!occluded&&!half_occluded&&detectWindow.width>1)
        {
            //no larger the search window????
            Rect myRect = detectWindow;

            cv::rectangle(Color, myRect, cv::Scalar(250*(1), 250*(1-1), 250*(1-2)), 2, CV_AA);
            //            cv::imshow( "show_search_window", Color );
            //            waitKey(2);

            cv::Mat roi_depth(Depth, myRect);
            cv::calcHist(&roi_depth, 1, 0, Mat(),tmp_depth_hist_pdf, 1, &d_bins, &pd_ranges);
            //            cout<<"tmp_depth_hist_pdf"<<tmp_depth_hist_pdf<<endl<<endl;
            tmp_depth_hist_pdf.at<float>(0,0)=0;
            std::vector<Point> peaks;
            findHistPeaks(tmp_depth_hist_pdf, peaks);


            //regard every peak as a point (depth,height of this depth in histogram)
            //regrad the last match peak also as a point
            cv::Point last_match_peak(depth_mu,depth_mu_value);
            peaks.push_back(last_match_peak);

            double peaks_mu=0.0;
            double peaks_mu_value=0.0;
            for(std::vector<Point>::iterator it= peaks.begin();it!=peaks.end();it++)
            {
                peaks_mu=peaks_mu+(*it).y;
                peaks_mu_value=peaks_mu_value+(*it).x;
            }
            peaks_mu=peaks_mu/peaks.size();
            peaks_mu_value=peaks_mu_value/peaks.size();

            double peaks_sigma=0.0;
            double peaks_sigma_value=0.0;
            for(std::vector<Point>::iterator it= peaks.begin();it!=peaks.end();it++)
            {
                peaks_sigma=peaks_sigma+((*it).y-peaks_mu)*((*it).y-peaks_mu);
                peaks_sigma_value=peaks_sigma_value+((*it).x-peaks_mu_value)*((*it).x-peaks_mu_value);
            }
            peaks_sigma=sqrt(peaks_sigma/peaks.size());
            peaks_sigma_value=sqrt(peaks_sigma_value/peaks.size());


            //nomalised distance (p-mu)/sigma
            double depth_mu_old_n=(depth_mu-peaks_mu)/peaks_sigma;
            double depth_mu_value_old_n=(depth_mu_value-peaks_mu_value)/peaks_sigma_value;

            peaks.pop_back();
            double normalize_distance=65535;
            for(std::vector<Point>::iterator it= peaks.begin();it!=peaks.end();it++)
            {
                double peak_mu_n=((*it).y-peaks_mu)/peaks_sigma;
                double peak_mu_value_n=((*it).x-peaks_mu_value)/peaks_sigma_value;
                double _normalize_distance=sqrt(pow((peak_mu_n-depth_mu_old_n),2)+pow((peak_mu_value_n-depth_mu_value_old_n),2));
                if(_normalize_distance<normalize_distance)
                {
                    normalize_distance=_normalize_distance;
                    depth_mu=(*it).y;
                    depth_mu_value=(*it).x;
                }
            }






            int it_start=(depth_mu>251)?251:(depth_mu+4);
            tmp_depth_hist_pdf.rowRange(it_start,255)=0;
            //cout<<tmp_depth_hist_pdf<<endl<<endl;
            tmp_depth_hist_pdf=tmp_depth_hist_pdf/sum(tmp_depth_hist_pdf)[0];
            cv::normalize(tmp_depth_hist_pdf, tmp_depth_hist_pdf, 0, 255, CV_MINMAX);


            //            auto start = std::chrono::high_resolution_clock::now();

            //creat a search window and do the backproj
            detectWindow=detectWindow&Rect(0,0,Color.size().width,Color.size().height);
            int detectWindow_XL=detectWindow.x-AREA_TOLERANCE,detectWindow_XR=detectWindow.x+detectWindow.width+AREA_TOLERANCE;
            int detectWindow_YT=detectWindow.y-AREA_TOLERANCE,detectWindow_YB=detectWindow.y+detectWindow.height+AREA_TOLERANCE;
            Rect search_window=Rect(detectWindow_XL,detectWindow_YT,detectWindow_XR-detectWindow_XL,detectWindow_YB-detectWindow_YT)&Rect(0,0,Color.size().width,Color.size().height);

            Mat hsv_search_window=Color(search_window);
            Mat depth_search_window=Depth(search_window);
            Mat backproj_search_window;


            //hsv backproj
            cv::calcBackProject( &hsv_search_window, 1, hsv_channels, hsv_hist, backproj_search_window, phsv_ranges, 1, true );
            //            imshow("backproj_tmp_color",backproj_search_window);


            //depth backproj
            Mat backproj_tmp_depth;
            cv::calcBackProject( &depth_search_window, 1, 0, tmp_depth_hist_pdf, backproj_tmp_depth, &pd_ranges, 1, true );
            //                                    imshow("backproj_tmp_depth",backproj_tmp_depth);

            //joint porj
            backproj_search_window.convertTo(backproj_search_window,CV_16U);
            backproj_tmp_depth.convertTo(backproj_tmp_depth,CV_16U);
            backproj_search_window=backproj_search_window.mul(backproj_tmp_depth);
            backproj_search_window.convertTo(backproj_search_window, CV_8U,255.0/65536.0,0);



            backproj=Mat::zeros(Color.size(),CV_8UC1);
            backproj_search_window.copyTo(backproj(search_window));

            //                        imshow("hsvd_back",backproj);
            //                        waitKey(1);
            //            auto now = std::chrono::high_resolution_clock::now();
            //            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            //            std::cout<<elapsed<<std::endl;
            occ_frames=0;
        }

        else{
            //            depth_mu=0;
            //            depth_mu_value=0;
            //            cv::calcBackProject( &Color, 1, hsv_channels, hsv_hist, backproj, phsv_ranges, 1, true );
            //            threshold( backproj, backproj, 5, 255,THRESH_TOZERO);


            occ_frames++;
            Mat depth_mask_occ;
            cv::inRange(Depth, cv::Scalar(depth_mu-5-occ_frames),cv::Scalar(depth_mu+5+occ_frames), depth_mask_occ);
            cv::calcBackProject( &Color, 1, hsv_channels, hsv_hist, backproj, phsv_ranges, 1, true );
            //            backproj=backproj&depth_mask_occ;
            //            threshold( backproj, backproj, 5, 255,THRESH_TOZERO);

            //            imshow("hsvd_back",backproj);
            //            imwrite("/home/zhao/backproj_color.png",backproj);
            //            imwrite("/home/zhao/color.png",Color);

        }
    }

}

void Object_Detector::RGBD_backprojection()
{
    const int hsv_size[] = { 16, 16 ,16};

    float h_range[] = { 0, 255};
    float s_range[] = { 0, 255 };//the upper boundary is exclusive
    float v_range[] = { 0, 255};
    float d_range[] = { (float)DMin, (float)DMax};

    const float* pd_ranges= d_range;
    const float* phsv_ranges[] = {h_range, s_range ,v_range };

    int hsv_channels[] = {0,1,2};

    if( firstRun )
    {
        //        depth_mu=255;
        //        depth_sigma=5;

        //        Mat color_rgb;
        //        cvtColor(Color,color_rgb,CV_HSV2BGR);

        //        cv::rectangle(color_rgb, selection, cv::Scalar(250*(1), 250*(1-1), 250*(1-2)), 2, CV_AA);
        //        cv::imshow( "show_search_window", Color );
        //        waitKey(2);

        cv::Mat roi_depth(Depth, selection);
        //        cv::Mat depth_mask_=Mat::ones(Color.size(),CV_8UC1);
        //        cv::Mat  depth_mask_roi(depth_mask_, selection);
        cv::calcHist(&roi_depth, 1, 0, cv::Mat(),tmp_depth_hist_pdf, 1, &d_bins, &pd_ranges);
        tmp_depth_hist_pdf.at<float>(0,0)=0;
        tmp_depth_hist_pdf.at<float>(0,255)=0;

        std::vector<Point> peaks;
        findHistPeaks(tmp_depth_hist_pdf, peaks);
        for(std::vector<Point>::iterator it= peaks.begin();it!=peaks.end();it++)
        {
            if(depth_mu>(*it).y&&(*it).y>5)
            {
                depth_mu=(*it).y;
                depth_mu_value=(*it).x;
            }
        }

        Mat tmp_depth_hist_pdf_=Mat::zeros(tmp_depth_hist_pdf.size(),tmp_depth_hist_pdf.type());
        //        int it_start=(depth_mu>depth_sigma)?(depth_mu-depth_sigma):0;

        for(int i =0;i<depth_mu+3;i++)
        {
            tmp_depth_hist_pdf_.at<float>(i,1)=tmp_depth_hist_pdf.at<float>(i,1);
        }
        double sum_tmp_depth_hist_pdf=sum(tmp_depth_hist_pdf_)[0];
        tmp_depth_hist_pdf=tmp_depth_hist_pdf_/sum_tmp_depth_hist_pdf;

        cv::Mat roi(Color, selection);
        Mat depth_roi=Depth(selection),depth_roi_mask;
        //       imshow("depth_roi",depth_roi);

        //        waitKey(100000);
        int _depth_sigma=0;

        for(; _depth_sigma<10;_depth_sigma++)
        {

            cv::inRange(depth_roi, cv::Scalar(0),cv::Scalar(depth_mu+_depth_sigma), depth_roi_mask);
            int count_white = cv::countNonZero(depth_roi_mask == 255);
            std::cout<<"(float(count_white))/(selection.area())"<<(float(count_white))/(selection.area())<<std::endl;
            if((float(count_white))/(selection.area())>0.6)
                break;
        }


        //        cv::inRange(Depth, cv::Scalar(0),cv::Scalar(depth_mu+_depth_sigma), depth_mask);
        //        imshow("depth_mask",depth_mask);


        Mat maskroi= depth_roi_mask;
        Mat check_roi;
        roi.copyTo(check_roi,maskroi);
        imshow("check_roi",check_roi);

        waitKey(1);
        ///
        ///
        //use one line to calculate the hsv_hist
        cv::calcHist(&roi, 1, hsv_channels, maskroi, hsv_hist, 3, hsv_size, phsv_ranges, true, false);
        //        cv::calcHist(&roi, 1, hsv_channels, Mat(), hsv_hist, 3, hsv_size, phsv_ranges, true, false);

        //use loop to calculate the hsv)hist
        //                cv::calcHist(&roi, 1, hs_channels, maskroi, hs_hist_for_HSV, 2, hs_size, phs_ranges, true, false);
        //                cv::normalize(hs_hist_for_HSV , hs_hist_for_HSV , 0, 255, CV_MINMAX);

        //                double sum_hs_hist=sum(hs_hist_for_HSV)[0];
        //                hs_hist_pdf=hs_hist_for_HSV/sum_hs_hist;

        //                cv::calcHist(&roi, 1, hsv_channels, maskroi, hsv_hist, 3, hsv_size, phsv_ranges, true, false);

        //                cv::calcHist(&roi, 1, v_channels, maskroi, tmp_value_hist_pdf, 1, &v_bins, &pv_ranges);
        //                double sum_tmp_value_hist_pdf=sum(tmp_value_hist_pdf)[0];
        //                tmp_value_hist_pdf=tmp_value_hist_pdf/sum_tmp_value_hist_pdf;

        //                for (int i=0; i<hsv_size[0]; i++) {
        //                    for (int j=0; j<hsv_size[1]; j++) {
        //                        for (int k=0; k<hsv_size[2]; k++) {
        //                            hsv_hist.at<float>(i,j,k)=255*hs_hist_pdf.at<float>(i,j)*tmp_value_hist_pdf.at<float>(k);
        //                        }
        //                    }
        //                }
        //        hsv_hist=hsv_hist/sum(hsv_hist)[0];



        //calculate the surrounding hist

        ///  ///  ///  ///use the surrounding/background to weight the color model
        ///
        cv::Point inflationPoint(-(selection.width/2), -(selection.height/2));
        cv::Size inflationSize(selection.width, selection.height);
        //        cv::Point inflationPoint(-(selection.width), -(selection.height));
        //        cv::Size inflationSize(selection.width*2, selection.height*2);
        Rect surrounding_rect = selection+inflationPoint;
        surrounding_rect += inflationSize;
        surrounding_rect=surrounding_rect&Rect(0,0,Color.size().width,Color.size().height);

        ////think twice about this , try not to weaker the target model,
        /// combine two bg model???????????????????????????????
        cv::Mat s_roi(Color, surrounding_rect);

        //        cv::inRange(Depth, cv::Scalar(0),cv::Scalar(depth_mu+_depth_sigma), depth_mask);
        //        Mat inv_depth_mask=255-depth_mask;

        Mat inv_depth_mask=Mat::zeros(Color.size(),CV_8UC1);
        Mat _inv_depth_mask=255*Mat::ones(Color.size(),CV_8UC1);
        inv_depth_mask(selection).copyTo(_inv_depth_mask(selection));
        //        imshow("_inv_depth_mask",_inv_depth_mask);
        cv::Mat s_maskroi(_inv_depth_mask, surrounding_rect);

        //use one line to calculate the s-hsv_hist
        cv::calcHist(&s_roi, 1, hsv_channels, s_maskroi, s_hsv_hist, 3, hsv_size, phsv_ranges, true, false);


        //                //use loop to calculate the hs*v hist
        //                Mat _tmp_hs_hist;
        //                cv::calcHist(&s_roi, 1, hs_channels, s_maskroi, _tmp_hs_hist, 2, hs_size, phs_ranges, true, false);
        //                cv::normalize(_tmp_hs_hist , _tmp_hs_hist , 0, 255, CV_MINMAX);
        //                double _sum_hs_hist=sum(_tmp_hs_hist)[0];
        //                Mat _tmp_hs_hist_pdf;
        //                _tmp_hs_hist_pdf=_tmp_hs_hist/_sum_hs_hist;
        //                Mat _tmp_value_hist;
        //                cv::calcHist(&s_roi, 1, v_channels, s_maskroi, _tmp_value_hist, 1, &v_bins, &pv_ranges);
        //                double _sum_tmp_value_hist=sum(_tmp_value_hist)[0];
        //                Mat _tmp_value_hist_pdf;
        //                _tmp_value_hist_pdf=_tmp_value_hist/_sum_tmp_value_hist;
        //                for (int i=0; i<hsv_size[0]; i++) {
        //                    for (int j=0; j<hsv_size[1]; j++) {
        //                        for (int k=0; k<hsv_size[2]; k++) {
        //                            s_hsv_hist.at<float>(i,j,k)=255*_tmp_hs_hist_pdf.at<float>(i,j)*_tmp_value_hist_pdf.at<float>(k);
        //                        }
        //                    }
        //                }

        //        s_hsv_hist=s_hsv_hist/sum(s_hsv_hist)[0];//no need to do this,


        double s_hsv_hist_min,s_hsv_hist_max;
        Mat s_hsv_hist_mask_nonzero = s_hsv_hist>0;
        cv::minMaxIdx(s_hsv_hist, &s_hsv_hist_min, &s_hsv_hist_max,NULL,NULL,s_hsv_hist_mask_nonzero);
        std::cout<<"s_hsv_hist_min: "<<s_hsv_hist_min<<std::endl;
        //        double sum = 0;
        //        for (int i=0; i<hsv_size[0]; i++) {
        //            for (int j=0; j<hsv_size[1]; j++) {
        //                for (int k=0; k<hsv_size[2]; k++) {

        //                    double background_weight;
        //                    if(s_hsv_hist.at<float>(i,j,k)>0)
        //                        background_weight=(s_hsv_hist_min)/(s_hsv_hist.at<float>(i,j,k));
        //                    else
        //                        background_weight=1;

        //                    hsv_hist.at<float>(i,j,k)=(hsv_hist.at<float>(i,j,k))*background_weight;

        //                }
        //            }
        //        }

        /////the loop can be equal to
        ///
        s_hsv_hist.setTo(s_hsv_hist_min,s_hsv_hist==0);
        hsv_hist=hsv_hist*(s_hsv_hist_min)/(s_hsv_hist);
        //        hsv_hist=hsv_hist*(s_hsv_hist_min+1)/(s_hsv_hist+1);


        double hMin,hMax;
        minMaxIdx(hsv_hist, &hMin, &hMax);
        hsv_hist = 255 * hsv_hist / hMax;


        //        Mat hsv_hist_mask_low=hsv_hist>5;
        //        Mat _hsv_hist;
        //        hsv_hist.copyTo(_hsv_hist,hsv_hist_mask_low);
        //        hsv_hist=_hsv_hist;


        cv::calcBackProject( &Color, 1, hsv_channels, hsv_hist, backproj, phsv_ranges, 1, true );
        //        imshow("hsv_back",backproj);
        Mat backproj_tmp_depth;
        cv::normalize(tmp_depth_hist_pdf, tmp_depth_hist_pdf, 0, 255, CV_MINMAX);
        cv::calcBackProject( &Depth, 1, 0, tmp_depth_hist_pdf, backproj_tmp_depth, &pd_ranges, 1, true );
        //        imshow("backproj_tmp_depth",backproj_tmp_depth);
        backproj.convertTo(backproj,CV_16U);
        backproj_tmp_depth.convertTo(backproj_tmp_depth,CV_16U);
        backproj=backproj.mul(backproj_tmp_depth);
        backproj.convertTo(backproj, CV_8U,255.0/65536.0,0);

        detectWindow = selection;
        firstRun = false;

    }

    else
    {
        if(!occluded&&!half_occluded&&detectWindow.width>1)
        {
            //no larger the search window????
            Rect myRect = detectWindow;

            cv::rectangle(Color, myRect, cv::Scalar(250*(1), 250*(1-1), 250*(1-2)), 2, CV_AA);
            //            cv::imshow( "show_search_window", Color );
            //            waitKey(2);

            cv::Mat roi_depth(Depth, myRect);
            cv::calcHist(&roi_depth, 1, 0, Mat(),tmp_depth_hist_pdf, 1, &d_bins, &pd_ranges);
            //            cout<<"tmp_depth_hist_pdf"<<tmp_depth_hist_pdf<<endl<<endl;
            tmp_depth_hist_pdf.at<float>(0,0)=0;
            std::vector<Point> peaks;
            findHistPeaks(tmp_depth_hist_pdf, peaks);


            //regard every peak as a point (depth,height of this depth in histogram)
            //regrad the last match peak also as a point
            cv::Point last_match_peak(depth_mu,depth_mu_value);
            peaks.push_back(last_match_peak);

            double peaks_mu=0.0;
            double peaks_mu_value=0.0;
            for(std::vector<Point>::iterator it= peaks.begin();it!=peaks.end();it++)
            {
                peaks_mu=peaks_mu+(*it).y;
                peaks_mu_value=peaks_mu_value+(*it).x;
            }
            peaks_mu=peaks_mu/peaks.size();
            peaks_mu_value=peaks_mu_value/peaks.size();

            double peaks_sigma=0.0;
            double peaks_sigma_value=0.0;
            for(std::vector<Point>::iterator it= peaks.begin();it!=peaks.end();it++)
            {
                peaks_sigma=peaks_sigma+((*it).y-peaks_mu)*((*it).y-peaks_mu);
                peaks_sigma_value=peaks_sigma_value+((*it).x-peaks_mu_value)*((*it).x-peaks_mu_value);
            }
            peaks_sigma=sqrt(peaks_sigma/peaks.size());
            peaks_sigma_value=sqrt(peaks_sigma_value/peaks.size());


            //nomalised distance (p-mu)/sigma
            double depth_mu_old_n=(depth_mu-peaks_mu)/peaks_sigma;
            double depth_mu_value_old_n=(depth_mu_value-peaks_mu_value)/peaks_sigma_value;

            peaks.pop_back();
            double normalize_distance=65535;
            for(std::vector<Point>::iterator it= peaks.begin();it!=peaks.end();it++)
            {
                double peak_mu_n=((*it).y-peaks_mu)/peaks_sigma;
                double peak_mu_value_n=((*it).x-peaks_mu_value)/peaks_sigma_value;
                double _normalize_distance=sqrt(pow((peak_mu_n-depth_mu_old_n),2)+pow((peak_mu_value_n-depth_mu_value_old_n),2));
                if(_normalize_distance<normalize_distance)
                {
                    normalize_distance=_normalize_distance;
                    depth_mu=(*it).y;
                    depth_mu_value=(*it).x;
                }
            }






            int it_start=(depth_mu>251)?251:(depth_mu+4);
            tmp_depth_hist_pdf.rowRange(it_start,255)=0;
            //cout<<tmp_depth_hist_pdf<<endl<<endl;
            tmp_depth_hist_pdf=tmp_depth_hist_pdf/sum(tmp_depth_hist_pdf)[0];
            cv::normalize(tmp_depth_hist_pdf, tmp_depth_hist_pdf, 0, 255, CV_MINMAX);


            //            auto start = std::chrono::high_resolution_clock::now();

            //creat a search window and do the backproj
            detectWindow=detectWindow&Rect(0,0,Color.size().width,Color.size().height);
            int detectWindow_XL=detectWindow.x-AREA_TOLERANCE,detectWindow_XR=detectWindow.x+detectWindow.width+AREA_TOLERANCE;
            int detectWindow_YT=detectWindow.y-AREA_TOLERANCE,detectWindow_YB=detectWindow.y+detectWindow.height+AREA_TOLERANCE;
            Rect search_window=Rect(detectWindow_XL,detectWindow_YT,detectWindow_XR-detectWindow_XL,detectWindow_YB-detectWindow_YT)&Rect(0,0,Color.size().width,Color.size().height);

            Mat hsv_search_window=Color(search_window);
            Mat depth_search_window=Depth(search_window);
            Mat backproj_search_window;


            //hsv backproj
            cv::calcBackProject( &hsv_search_window, 1, hsv_channels, hsv_hist, backproj_search_window, phsv_ranges, 1, true );
            //                                    imshow("backproj_tmp_color",backproj_search_window);


            //depth backproj
            Mat backproj_tmp_depth;
            cv::calcBackProject( &depth_search_window, 1, 0, tmp_depth_hist_pdf, backproj_tmp_depth, &pd_ranges, 1, true );
            //                                    imshow("backproj_tmp_depth",backproj_tmp_depth);

            //joint porj
            backproj_search_window.convertTo(backproj_search_window,CV_16U);
            backproj_tmp_depth.convertTo(backproj_tmp_depth,CV_16U);
            backproj_search_window=backproj_search_window.mul(backproj_tmp_depth);
            backproj_search_window.convertTo(backproj_search_window, CV_8U,255.0/65536.0,0);



            backproj=Mat::zeros(Color.size(),CV_8UC1);
            backproj_search_window.copyTo(backproj(search_window));

            //                        imshow("hsvd_back",backproj);
            //                        waitKey(1);
            //            auto now = std::chrono::high_resolution_clock::now();
            //            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            //            std::cout<<elapsed<<std::endl;
            occ_frames=0;
        }

        else{
            //            depth_mu=0;
            //            depth_mu_value=0;
            //            cv::calcBackProject( &Color, 1, hsv_channels, hsv_hist, backproj, phsv_ranges, 1, true );
            //            threshold( backproj, backproj, 5, 255,THRESH_TOZERO);

            occ_frames++;
            Mat depth_mask_occ;
            cv::inRange(Depth, cv::Scalar(depth_mu-5-occ_frames),cv::Scalar(depth_mu+5+occ_frames), depth_mask_occ);
            cv::calcBackProject( &Color, 1, hsv_channels, hsv_hist, backproj, phsv_ranges, 1, true );
            backproj=backproj&depth_mask_occ;
            threshold( backproj, backproj, 5, 255,THRESH_TOZERO);

            //            imshow("hsvd_back",backproj);
            //            imwrite("/home/zhao/backproj_color.png",backproj);
            //            imwrite("/home/zhao/color.png",Color);

        }
    }

}


//camshift + occlusion handle
cv::RotatedRect Object_Detector::object_shift(InputArray _probColor,Rect& window, TermCriteria criteria)
{
    Size size;
    Mat mat;
    mat = _probColor.getMat(), size = mat.size();
    cv::meanShift( _probColor, window, criteria );

    //enlarge the window to use the cam-shift scale estimation
    //if drifted, update the window with the tracker msg
    if(with_occ_handle){
        if(drifted)
        {
            bool update_from_tracker=false;

            for (size_t i = 0; i < current_track2D_array.size(); ++i)
            {
                std::pair<std::string,cv::Point2d> track2D=current_track2D_array[i];

                if(track2D.first==object_name)
                {
                    update_from_tracker=true;

                    window.x=track2D.second.x-(last_tracked_window_without_occ.width);
                    window.y=track2D.second.y-(last_tracked_window_without_occ.height);
                    window.width =  last_tracked_window_without_occ.width*2;
                    window.height = last_tracked_window_without_occ.height*2;
                    window=window&Rect(0,0,size.width,size.height);
                    //std::cout<<"Update window from tracker feed back: "<< track2D.first<<" with the window: "<< window.x<<" "<< window.y<<" "<< window.width<<" "<<window.height<<std::endl;

                    ////update the "last_tracked_window_without_occ" with the center of the track,keep the old width and height reset the occ_frame.
                    /// or if occ for a long time, the distance between the old last_tracked_window_without_occ and the wrong detection maybe smaller than "occluded_frames*30"
                    last_tracked_window_without_occ.x=track2D.second.x;
                    last_tracked_window_without_occ.y=track2D.second.y;
                    occluded_frames=1;
                }
            }

            // if(!update_from_tracker)
            // std::cout<<"Trakcer drift and it failed to update window from tracker, the tracker msg size is : "<<current_track2D_array.size()<<std::endl;


        }

        else{
            window.x -= AREA_TOLERANCE;
            window.y -= AREA_TOLERANCE;
            window.width += 2 * AREA_TOLERANCE;
            window.height += 2 * AREA_TOLERANCE;
            window=window&Rect(0,0,size.width,size.height);
        }

    }
    else{
        window.x -= AREA_TOLERANCE;
        window.y -= AREA_TOLERANCE;
        window.width += 2 * AREA_TOLERANCE;
        window.height += 2 * AREA_TOLERANCE;
        window=window&Rect(0,0,size.width,size.height);}


    // Calculating moments in new center mass
    //  Moments m = isUMat ? moments(umat(window)) : moments(mat(window));
    Moments m = moments(mat(window));
    double m00 = m.m00, m10 = m.m10, m01 = m.m01;
    double mu11 = m.mu11, mu20 = m.mu20, mu02 = m.mu02;


    ////////////difference from cv::camshift//////////
    // occluded process
    if(with_occ_handle)
    {
        if( fabs(m00) < QUALITY_TOLERANCE ) //if totally occ
        {
            occluded=true;
            occluded_frames++;
            //            std::cout<<"totally occluded: window QUALITY:  "<<fabs(m00)<<std::endl;
            return RotatedRect();
        }

        else if(fabs(m00)/window.area()<DENSITY_TOLORENCE)// if half_occ
            //    else if(fabs(m_hsv_mask_window00)/window.area()<DENSITY_TOLORENCE)

        {
            if(drifted)// when difted,which means the window is updated from the tracker, the target can be recover in the right position with a bigger window, we need the Cam-shift scale to resize the window in the following frame rather than keep updating from traker all the time.
                drifted=false;

            occluded_frames++;
            occluded=true;
            half_occluded=true;
            //            std::cout<<"half occluded: window density "<<fabs(m00)/window.area()<<"  window quality:  "<<fabs(m00)<<std::endl;
            //            imwrite("/home/zhao/mat_window.png",mat(window));

            //half occlude means when the detecteor detected the similar background ,but the desity is not big enough ,or just a little part of the object appears
            //if there are a little similar thing in the background ,the detector will keep the search window around this similar thing and nolonger expand
            //in this situation ,if the real object appear ,it maybe not in the search window,do it will never been detect untill it move to the small search window
            //so ,we set half_occluded_frames=10,if the detector keep the search window around this similar thing for more than 10 frames,we will assume this thing is not the object and than expand the search window to the whole image
            //if this little similar thing is really belong to the object ,it will converged in 10 frames,so 10 is enough

            if(--half_occluded_frames==0)//if there are a little similar thing in the background ,the detector will keep the window around that ,but it's desity is not big enough so the detector will not asuue
            {
                half_occluded_frames=10;
                //                window=Rect(0,0,size.width,size.height);
                drifted=true; //if the tracker can't localize the target sprecisly in 10 frames, update from the tracker again.
                //          std::cout<<"set window to the whole image: half_occluded_frames: "<<half_occluded_frames<<std::endl;
                return RotatedRect();
            }
        }

        else{// if not occ(not_occ->not->occ, or occ->not_occ)
            if(occluded) //if occ in the last frame, not_occ in current frame, the trakcer has been recoverd from occ, set recover_from_occ=true , later to judge if the recover is correct.
            {
                recover_from_occ=true;
            }
            if(drifted) // difted==true means the window is updated from the tracker. In this window , the object is not occ. This means that the object is already shows up after occlusion.
                drifted=false;

            occluded=false;
            half_occluded=false;
            half_occluded_frames=10;
            //            std::cout<<"not occluded: window density "<<fabs(m00)/window.area()<<"  window quality:  "<<fabs(m00)<<std::endl;

        }
    }
    ////////////difference from cv::camshift////////////


    double inv_m00 = 1. / m00;
    int xc = cvRound( m10 * inv_m00 + window.x );
    int yc = cvRound( m01 * inv_m00 + window.y );
    double a = mu20 * inv_m00, b = mu11 * inv_m00, c = mu02 * inv_m00;

    // Calculating width & height
    double square = std::sqrt( 4 * b * b + (a - c) * (a - c) );

    // Calculating orientation
    double theta = atan2( 2 * b, a - c + square );

    // Calculating width & length of figure
    double cs = cos( theta );
    double sn = sin( theta );

    double rotate_a = cs * cs * mu20 + 2 * cs * sn * mu11 + sn * sn * mu02;
    double rotate_c = sn * sn * mu20 - 2 * cs * sn * mu11 + cs * cs * mu02;
    double length = std::sqrt( rotate_a * inv_m00 ) * 4;
    double width = std::sqrt( rotate_c * inv_m00 ) * 4;

    // In case, when tetta is 0 or 1.57... the Length & Width may be exchanged
    if( length < width )
    {
        std::swap( length, width );
        std::swap( cs, sn );
        theta = CV_PI*0.5 - theta;
    }


    // Saving results
    int _xc = cvRound( xc );
    int _yc = cvRound( yc );

    int t0 = cvRound( fabs( length * cs ));
    int t1 = cvRound( fabs( width * sn ));

    t0 = MAX( t0, t1 ) + 2;
    window.width = MIN( t0, (size.width - _xc) * 2 );

    t0 = cvRound( fabs( length * sn ));
    t1 = cvRound( fabs( width * cs ));

    t0 = MAX( t0, t1 ) + 2;
    window.height = MIN( t0, (size.height - _yc) * 2 );

    window.x = MAX( 0, _xc - window.width / 2 );
    window.y = MAX( 0, _yc - window.height / 2 );

    window.width = MIN( size.width - window.x, window.width );
    window.height = MIN( size.height - window.y, window.height );

    RotatedRect box;
    box.size.height = (float)length;
    box.size.width = (float)width;
    box.angle = (float)((CV_PI*0.5+theta)*180./CV_PI);
    while(box.angle < 0)
        box.angle += 360;
    while(box.angle >= 360)
        box.angle -= 360;
    if(box.angle >= 180)
        box.angle -= 180;
    box.center = Point2f( window.x + window.width*0.5f, window.y + window.height*0.5f);



    ////////////difference from cv::camshift////////////

    if(with_occ_handle){
        if(recover_from_occ){
            recover_from_occ=false;
            double window_distance_before_after_occ=sqrt(pow((window.x+(window.width/2)-(last_tracked_window_without_occ.x+last_tracked_window_without_occ.width/2)),2)+pow((window.y+window.height/2-(last_tracked_window_without_occ.y+last_tracked_window_without_occ.height/2)),2));
            //  std::cout<<"recover_from_occ and the distance is : "<<window_distance_before_after_occ <<" and the occ_frames is: "<<occluded_frames<<std::endl;

            if (window_distance_before_after_occ>occluded_frames*30)
            {
                drifted=true;
                //    std::cout<<"Tracker drifted..." <<std::endl;
                occluded_frames++;
                return RotatedRect();
            }
            else{
                //   std::cout<<"Recover sucessfully... "<<std::endl;
                occluded_frames=0;
                return box;
            }
        }
        else
            return box;
    }
    else
        return box;

    ////////////difference from cv::camshift////////////

}

cv::RotatedRect Object_Detector::detectCurrentRect(int id)
{

    mainColor.copyTo(Color);
    mainDepth.copyTo(Depth);

    //    cv::cvtColor(Color, hsv, CV_BGR2HSV);

    //calculate the hsv_mask by the range
    cv::inRange(Color, cv::Scalar(HMin, SMin, MIN(VMin,VMax)), cv::Scalar(HMax, SMax, MAX(VMin, VMax)), hsv_mask);

    if(Backprojection_Mode=="H")
    {
        H_backprojection();
    }
    else if(Backprojection_Mode=="HS")
    {
        HS_backprojection();
    }
    else if(Backprojection_Mode=="HSD")
    {
        HSD_backprojection();
    }
    else if(Backprojection_Mode=="HSV"){
        HSV_backprojection();
    }
    else if(Backprojection_Mode=="RGBD")
        RGBD_backprojection();
    else
        HSVD_backprojection(id);
    //    imshow("backproj first",backproj);


    // calculate the other_object_mask with the current_detected_boxes
    other_object_mask=Mat::ones(Color.size(), CV_8U)*255;
    for (int i=0; i<current_detected_boxes.size(); i++)
    {
        if(i!=id)
        {
            uchar tmp =0;
            Rect current_tracked_box =current_detected_boxes[i];
            current_tracked_box=current_tracked_box&Rect(0,0,Color.size().width,Color.size().height);
            other_object_mask(current_tracked_box)=tmp;
        }
    }




    if(Backprojection_Mode=="HSVD")
    {
        backproj &= other_object_mask;
    }
    else
    {
        detectWindow=detectWindow&Rect(0,0,Color.size().width,Color.size().height);
        if(occluded==false&&detectWindow.area()>1)
        {
            //use x y to generate position_mask,the object can move in the window which is AREA_TOLERANCE size bigger than the last detected window
            position_mask=Mat::zeros(Color.size(),CV_8UC1);
            int detectWindow_XL=detectWindow.x-AREA_TOLERANCE,detectWindow_XR=detectWindow.x+detectWindow.width+AREA_TOLERANCE;
            int detectWindow_YT=detectWindow.y-AREA_TOLERANCE,detectWindow_YB=detectWindow.y+detectWindow.height+AREA_TOLERANCE;
            Rect search_window=Rect(detectWindow_XL,detectWindow_YT,detectWindow_XR-detectWindow_XL,detectWindow_YB-detectWindow_YT)&Rect(0,0,Color.size().width,Color.size().height);
            position_mask(search_window)=255;
            position_mask &= hsv_mask;
            backproj &= position_mask;
            backproj &= other_object_mask;
        }
        else{
            backproj &= hsv_mask;
            backproj &= other_object_mask;
        }
    }



    if(detectWindow.area()>1)
        current_detectedBox = object_shift(backproj, detectWindow,
                                           cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
    if(!half_occluded&&!occluded&&!drifted)
        last_tracked_window_without_occ=current_detectedBox.boundingRect();
    //    std::cout<<"last_tracked_window_without_occ: "<<last_tracked_window_without_occ.x<<" "<<last_tracked_window_without_occ.y<<" "<<last_tracked_window_without_occ.width<<" "<<std::endl;
    //    cv::rectangle(backproj, detectWindow, cv::Scalar(255, 0, 0), 2, CV_AA);
    //    imshow("backproj final",backproj);
    //    cv::waitKey(1);
    return current_detectedBox;

}

