/*
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2011-2012, Matteo Munaro [matteo.munaro@dei.unipd.it], Filippo Basso [filippo.basso@dei.unipd.it]
 * Copyright (c) 2013-, Open Perception, Inc.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * * Neither the name of the copyright holder(s) nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Matteo Munaro [matteo.munaro@dei.unipd.it], Filippo Basso [filippo.basso@dei.unipd.it]
 *
 */

#include <opencv2/opencv.hpp>

#include <open_ptrack/tracking/tracker_object.h>

namespace open_ptrack
{
namespace tracking
{

TrackerObject::TrackerObject(
        double gate_distance,
        bool detector_likelihood,
        std::vector<double> likelihood_weights,
        bool velocity_in_motion_term,
        double min_confidence,
        double min_confidence_detections,
        double sec_before_old,
        double sec_before_fake,
        double sec_remain_new,
        int detections_to_validate,
        double period,
        double position_variance,
        double acceleration_variance,
        std::string world_frame_id,
        bool debug_mode,
        bool vertical) :
    gate_distance_(gate_distance),
    detector_likelihood_(detector_likelihood),
    likelihood_weights_(likelihood_weights),
    velocity_in_motion_term_(velocity_in_motion_term),
    min_confidence_(min_confidence),
    min_confidence_detections_(min_confidence_detections),
    sec_before_old_(sec_before_old),
    sec_before_fake_(sec_before_fake),
    sec_remain_new_(sec_remain_new),
    detections_to_validate_(detections_to_validate),
    period_(period),
    position_variance_(position_variance),
    acceleration_variance_(acceleration_variance),
    world_frame_id_(world_frame_id),
    debug_mode_(debug_mode),
    vertical_(vertical)
{
    tracks_counter_ = 0;
    trackers_initialized=false;

}

TrackerObject::~TrackerObject()
{
    // TODO Auto-generated destructor stub
}

void
TrackerObject::newFrame(const std::vector<open_ptrack::detection::Detection>& detections)
{
    detections_.clear();
    unassociated_detections_.clear();
    lost_tracks_.clear();
    new_tracks_.clear();
    detections_ = detections;
    ros::Time current_detections_time = detections_[0].getSource()->getTime();

    for(std::list<open_ptrack::tracking::TrackObject*>::iterator it = tracks_.begin(); it != tracks_.end();)
    {
        open_ptrack::tracking::TrackObject* t = *it;
        bool deleted = false;

        if(((t->getVisibility() == TrackObject::NOT_VISIBLE && (t->getSecFromLastHighConfidenceDetection(current_detections_time)) >= sec_before_old_)
            || (!t->isValidated() && t->getSecFromFirstDetection(current_detections_time) >= sec_before_fake_)))
        {
            if (debug_mode_)
            {
                std::cout << "Track " << t->getId() << " DELETED" << std::endl;
            }
            delete t;
            it = tracks_.erase(it);
            deleted = true;
        }
        else if(!t->isValidated() && t->getUpdatesWithEnoughConfidence() == detections_to_validate_)
        {
            t->validate();
            if (debug_mode_)
            {
                std::cout << "Track " << t->getId() << " VALIDATED" << std::endl;
            }
        }
        else if(t->getStatus() == TrackObject::NEW && t->getSecFromFirstDetection(current_detections_time) >= sec_remain_new_)
        {
            t->setStatus(TrackObject::NORMAL);
            if (debug_mode_)
            {
                std::cout << "Track " << t->getId() << " set to NORMAL" << std::endl;
            }
        }


        if(!deleted)
        {
            if(t->getStatus() == TrackObject::NEW && t->getVisibility() == TrackObject::VISIBLE)
                new_tracks_.push_back(t);
            if(t->getVisibility() == TrackObject::NOT_VISIBLE)
                lost_tracks_.push_back(t);
            it++;
        }
    }
}

void
TrackerObject::updateTracks()
{
    createDistanceMatrix();
    createCostMatrix();

    // Solve Global Nearest Neighbor problem:
    Munkres munkres;
    cost_matrix_ = munkres.solve(cost_matrix_, false);	// rows: targets (tracks), cols: detections

    updateDetectedTracks();
    if(!trackers_initialized){
        fillUnassociatedDetections();
        trackers_initialized=true;
    }

    updateLostTracks();
    createNewTracks();
}

void
TrackerObject::toMarkerArray(visualization_msgs::MarkerArray::Ptr& msg,Eigen::Vector3f gt1,Eigen::Vector3f gt2)
{
    for(std::list<open_ptrack::tracking::TrackObject*>::iterator it =
        tracks_.begin(); it != tracks_.end(); it++)
    {
        open_ptrack::tracking::TrackObject* t = *it;
        t->createMarker(msg,gt1, gt2);
    }
}

void
TrackerObject::toMarkerArray(visualization_msgs::MarkerArray::Ptr& msg,std::vector<Eigen::Vector3f> gts)
{
    for(std::list<open_ptrack::tracking::TrackObject*>::iterator it =
        tracks_.begin(); it != tracks_.end(); it++)
    {
        open_ptrack::tracking::TrackObject* t = *it;
        t->createMarker(msg,gts);
    }
}

void
TrackerObject::toMsg(opt_msgs::Track3DArray::Ptr& msg)
{
    for(std::list<open_ptrack::tracking::TrackObject*>::iterator it =
        tracks_.begin(); it != tracks_.end(); it++)
    {
        open_ptrack::tracking::TrackObject* t = *it;

        opt_msgs::Track3D track;
        t->toMsg(track, vertical_);
        msg->tracks.push_back(track);
    }
}

void
TrackerObject::toMsg(opt_msgs::Track3DArray::Ptr& msg,
                     std::string& source_frame_id)
{
    for(std::list<open_ptrack::tracking::TrackObject*>::iterator it =
        tracks_.begin(); it != tracks_.end(); it++)
    {
        open_ptrack::tracking::TrackObject* t = *it;
        if (strcmp(t->getDetectionSource()->getFrameId().c_str(),
                   source_frame_id.c_str()) == 0) // if strings are equal
        {
            opt_msgs::Track3D track;
            t->toMsg(track, vertical_);

            //          // For publishing only not occluded tracks:
            //          if (track.visibility < 2)
            //            msg->tracks.push_back(track);

            // For publishing all tracks:
            msg->tracks.push_back(track);
        }
    }
}



void
TrackerObject::to_object_name_Msg(opt_msgs::ObjectNameArray::Ptr& msg)
{
    if(association_for_initialize_objectnames_.size()==tracks_.size())
    {
        int index=0;
        for(std::list<open_ptrack::tracking::TrackObject*>::iterator it = tracks_.begin(); it != tracks_.end(); it++)
        {
            open_ptrack::tracking::TrackObject* t = *it;
            opt_msgs::ObjectName object_name_msg;
            object_name_msg.object_name=t->object_name_;
            object_name_msg.no=association_for_initialize_objectnames_[index];
            msg->object_names.push_back(object_name_msg);
            index++;
            //      std::cout<<"send message with object name:"<<t->object_name_<<std::endl;
        }
    }
    association_for_initialize_objectnames_.clear();
}



void
TrackerObject::getAliveIDs (opt_msgs::IDArray::Ptr& msg)
{
    for(std::list<open_ptrack::tracking::TrackObject*>::iterator it = tracks_.begin(); it != tracks_.end(); it++)
    {
        open_ptrack::tracking::TrackObject* t = *it;
        msg->ids.push_back ((*it)->getId());
    }
    msg->max_ID = tracks_counter_;
}

size_t
TrackerObject::appendToPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointcloud, size_t starting_index, size_t max_size)
{
    for(size_t i = 0; i < tracks_.size() && pointcloud->size() < max_size; i++)
    {
        pcl::PointXYZRGB point;
        pointcloud->push_back(point);
    }

    for(std::list<open_ptrack::tracking::TrackObject*>::iterator it = tracks_.begin(); it != tracks_.end(); it++)
    {
        open_ptrack::tracking::TrackObject* t = *it;
        if(t->getPointXYZRGB(pointcloud->points[starting_index]))
            starting_index = (starting_index + 1) % max_size;

    }
    return starting_index;
}

/************************ protected methods ************************/

int
TrackerObject::createNewTrack(open_ptrack::detection::Detection& detection)
{
    if(detection.getConfidence() < min_confidence_)
        return -1;

    open_ptrack::tracking::TrackObject* t;
    t = new open_ptrack::tracking::TrackObject(
                ++tracks_counter_,
                world_frame_id_,
                position_variance_,
                acceleration_variance_,
                period_,
                velocity_in_motion_term_  );


    t->init(detection.getWorldCentroid()(0), detection.getWorldCentroid()(1),detection.getWorldCentroid()(2),
            detection.getHeight(), detection.getDistance(), detection.getObjectName(),detection.getSource());

    bool first_update = true;
    t->update(detection.getWorldCentroid()(0), detection.getWorldCentroid()(1), detection.getWorldCentroid()(2),
              detection.getHeight(), detection.getDistance(), detection.getObjectName(),0.0,
              detection.getConfidence(), min_confidence_, min_confidence_detections_, detection.getSource(), first_update);

    ROS_INFO("Created %d", t->getId());

    tracks_.push_back(t);
    return tracks_counter_;
}
int
TrackerObject::createNewTrack(open_ptrack::detection::Detection& detection, bool distract_track)
{
    if(detection.getConfidence() < min_confidence_)
        return -1;

    open_ptrack::tracking::TrackObject* t;
    t = new open_ptrack::tracking::TrackObject(
                ++tracks_counter_,
                world_frame_id_,
                position_variance_,
                acceleration_variance_,
                period_,
                velocity_in_motion_term_  );
    if(!distract_track)
    {
        t->init(detection.getWorldCentroid()(0), detection.getWorldCentroid()(1),detection.getWorldCentroid()(2),
                detection.getHeight(), detection.getDistance(), detection.getObjectName(),detection.getSource());

        bool first_update = true;
        t->update(detection.getWorldCentroid()(0), detection.getWorldCentroid()(1), detection.getWorldCentroid()(2),
                  detection.getHeight(), detection.getDistance(), detection.getObjectName(),0.0,
                  detection.getConfidence(), min_confidence_, min_confidence_detections_, detection.getSource(), first_update);
    }
    else{
        std::string tracker_name=detection.getObjectName()+"_distract_track";
        t->init(detection.getWorldCentroid()(0), detection.getWorldCentroid()(1),detection.getWorldCentroid()(2),
                detection.getHeight(), detection.getDistance(), tracker_name,detection.getSource());

        bool first_update = true;
        t->update(detection.getWorldCentroid()(0), detection.getWorldCentroid()(1), detection.getWorldCentroid()(2),
                  detection.getHeight(), detection.getDistance(), tracker_name,0.0,
                  detection.getConfidence(), min_confidence_, min_confidence_detections_, detection.getSource(), first_update);
    }
    ROS_INFO("Created %d", t->getId());

    tracks_.push_back(t);
    return tracks_counter_;
}

void
TrackerObject::createDistanceMatrix()
{
    distance_matrix_ = cv::Mat_<double>(tracks_.size(), detections_.size());
    int track = 0;
    for(std::list<TrackObject*>::iterator it = tracks_.begin(); it != tracks_.end(); it++)
    {
        //double x, y, height, vx, vz;
        TrackObject* t = *it;
        //t->predict(x, y, height, vx, vz);
        int measure = 0;
        std::string track_object_name=t->object_name_;

        for(std::vector<open_ptrack::detection::Detection>::iterator dit = detections_.begin(); dit != detections_.end(); dit++)
        {

            //for same object with diff name, should be conmmented
//            std::string detection_object_name= dit->getObjectName();
//            if (detection_object_name.compare("default")!=0 && detection_object_name.compare(track_object_name)!=0 && track_object_name.compare("default")!=0)
//            {
//                distance_matrix_(track, measure++)=2*gate_distance_;
//            }
//            else
//            {
            //for same object with diff name, should be conmmented


                double detector_likelihood;
                // Compute detector likelihood:
                if (detector_likelihood_)
                {
                    detector_likelihood = dit->getConfidence();
                    //				detector_likelihood = log((dit->getConfidence() + 3) / 6);
                }
                else
                {
                    detector_likelihood = 0;
                }

                // Compute motion likelihood:
                double motion_likelihood = t->getMahalanobisDistance(
                            dit->getWorldCentroid()(0),
                            dit->getWorldCentroid()(1),
                            dit->getWorldCentroid()(2),
                            dit->getSource()->getTime());

                // Compute joint likelihood and put it in the distance matrix:

                distance_matrix_(track, measure++) = likelihood_weights_[0] * detector_likelihood + likelihood_weights_[1] * motion_likelihood;

                // Remove NaN and inf:
                if (std::isnan(distance_matrix_(track, measure-1)) | (not std::isfinite(distance_matrix_(track, measure-1))))
                    distance_matrix_(track, measure-1) = 2*gate_distance_;

                //          std::cout << (*it)->getId() << ": " << "Motion likelihood: " << likelihood_weights_[0] * motion_likelihood << std::endl;
                //          if (detector_likelihood_)
                //          	std::cout << (*it)->getId() << ": " << "Detector likelihood: " << likelihood_weights_[1] * dit->getConfidence() << std::endl;
                //          std::cout << (*it)->getId() << ": " << "JOINT LIKELIHOOD: " << distance_matrix_(track, measure-1) << std::endl;

                /*ROS_INFO("%d(%f, %f) = %f", t->getId(),
          dit->getWorldCentroid()(0),
          dit->getWorldCentroid()(1),
          distance_matrix_(track, measure - 1));*/

            //for same object with diff name, should be conmmented
//            }
            //for same object with diff name, should be conmmented

        }
        track++;
    }

    //      	std::cout << "Distance matrix:" << std::endl;
    //      	for(int row = 0; row < distance_matrix_.rows; row++)
    //      	{
    //      		for(int col = 0; col < distance_matrix_.cols; col++)
    //      		{
    //      			std::cout.width(8);
    //      			std::cout << distance_matrix_(row,col) << ",";
    //      		}
    //      		std::cout << std::endl;
    //      	}
    //      	std::cout << std::endl;
}

void
TrackerObject::createCostMatrix()
{
    cost_matrix_ = distance_matrix_.clone();
    for(int i = 0; i < distance_matrix_.rows; i++)
    {
        for(int j = 0; j < distance_matrix_.cols; j++)
        {
            if(distance_matrix_(i, j) > gate_distance_)
                cost_matrix_(i, j) = 1000000.0;
        }
    }


    //      	std::cout << "Munkres input matrix:" << std::endl;
    //      	for(int row = 0; row < cost_matrix_.rows; row++)
    //      	{
    //      		for(int col = 0; col < cost_matrix_.cols; col++)
    //      		{
    //      			std::cout.width(8);
    //      			std::cout << cost_matrix_(row,col) << ",";
    //      		}
    //      		std::cout << std::endl;
    //      	}
    //      	std::cout << std::endl;
}

void
TrackerObject::updateDetectedTracks()
{
    //      	std::cout << "Munkres output matrix:" << std::endl;
    //      	for(int row = 0; row < cosyt_matrix_.rows; row++) {
    //      		for(int col = 0; col < cost_matrix_.cols; col++) {
    //      			std::cout.width(1);
    //      			std::cout << cost_matrix_(row,col) << ",";
    //      		}
    //      		std::cout << std::endl;
    //      	}
    //      	std::cout << std::endl;

    // Iterate over every track:
    int track = 0;
    for(std::list<open_ptrack::tracking::TrackObject*>::iterator it = tracks_.begin(); it != tracks_.end(); it++)
    {
        bool updated = false;
        open_ptrack::tracking::TrackObject* t = *it;

        for(int measure = 0; measure < cost_matrix_.cols; measure++)
        {
            // If a detection<->track association has been found:
            if(cost_matrix_(track, measure) == 0.0 && distance_matrix_(track, measure) <= gate_distance_)
            {


                open_ptrack::detection::Detection& d = detections_[measure];

                // If the detection has enough confidence in the current frame or in a recent past:
                //            if ((t->getLowConfidenceConsecutiveFrames() < 10) || ((d.getConfidence() - 0.5) > min_confidence_detections_))
                if ((t->getLowConfidenceConsecutiveFrames() < 10) || (d.getConfidence() > ((min_confidence_ + min_confidence_detections_)/2)))
                {
                    //for object tracking
                    association_for_initialize_objectnames_.push_back(measure);


                    // Update track with the associated detection:
                    bool first_update = false;

                    t->update(d.getWorldCentroid()(0), d.getWorldCentroid()(1), d.getWorldCentroid()(2),d.getHeight(),
                              d.getDistance(),d.getObjectName(), distance_matrix_(track, measure),
                              d.getConfidence(), min_confidence_, min_confidence_detections_,
                              d.getSource(), first_update);

                    t->setVisibility(d.isOccluded() ? TrackObject::OCCLUDED : TrackObject::VISIBLE);
                    updated = true;
                    break;
                }
                else
                {
                    //std::cout << "Id: " << t->getId() << ", lowConfConsFrames: " << t->getLowConfidenceConsecutiveFrames() << ", newConf: " << d.getConfidence()<< std::endl;
                }
            }
        }
        if(!updated)
        {
            if(t->getVisibility() != TrackObject::NOT_VISIBLE)
            {
                t->setVisibility(TrackObject::NOT_VISIBLE);
                //t->update();
            }
        }
        track++;
    }
    //	std::cout << std::endl;
}

void
TrackerObject::fillUnassociatedDetections()
{
    // Fill a list with detections not associated to any track:
    if(cost_matrix_.cols == 0 && detections_.size() > 0)
    {
        for(size_t measure = 0; measure < detections_.size(); measure++)
            unassociated_detections_.push_back(detections_[measure]);
    }
    else
    {
        for(int measure = 0; measure < cost_matrix_.cols; measure++)
        {
            bool associated = false;
            for(int track = 0; track < cost_matrix_.rows; track++)
            {
                if(cost_matrix_(track, measure) == 0.0)
                {
                    if(distance_matrix_(track, measure) > gate_distance_)
                        break;
                    associated = true;
                }
            }
            if(!associated/* && detections_[measure].getConfidence() > min_confidence_*/)
            {
                unassociated_detections_.push_back(detections_[measure]);
            }
        }
    }
}

void
TrackerObject::updateLostTracks()
{
    //for(std::list<open_ptrack::tracking::Track*>::iterator it = lost_tracks_.begin(); it != lost_tracks_.end(); it++)
    //	(*it)->update();
}

void
TrackerObject::createNewTracks()
{
    for(std::list<open_ptrack::detection::Detection>::iterator dit = unassociated_detections_.begin();
        dit != unassociated_detections_.end(); dit++)
    {
        createNewTrack(*dit);
        //        open_ptrack::detection::Detection d=*dit;
        //         Eigen::Vector3d _detection_position=d.getWorldCentroid();
        //        std::cout<<"create new track_detection_position x, y, z"<<_detection_position[0]<<" "<<_detection_position[1]<<" "<<_detection_position[2]<<std::endl;

        /*


        open_ptrack::detection::Detection d=*dit;
        bool distract_detection=false;

        std::string  unassociated_detection_object_name=d.getObjectName();
        Eigen::Vector3d detection_position=d.getWorldCentroid();
        ros::Time unassociated_detection_time = d.getSource()->getTime();

        for(std::list<open_ptrack::tracking::TrackObject*>::iterator it = tracks_.begin(); it != tracks_.end(); it++)
        {
            TrackObject* t = *it;
            std::string track_object_name=t->object_name_;
            //            std::cout<<"track id: "<<t->getId()<<"track_visible: "<<t->getVisibility()<<"  track name: "<<track_object_name<<" unassociated_object_name"<<unassociated_detection_object_name<<std::endl;
            //            if(t->getStatus()==TrackObject::NORMAL&& t->isValidated())
            //            {
            //                double track_filter_x,track_filter_y,track_filter_z;
            //                t->getState(track_filter_x, track_filter_y, track_filter_z);
            //                double distance= sqrt(pow((track_filter_x-detection_position[0]),2)+pow((track_filter_y-detection_position[1]),2)+pow((track_filter_z-detection_position[2]),2));
            //                float time_diff=t->getSecFromLastHighConfidenceDetection(unassociated_detection_time);
            ////                std::cout<<"track id:"<<t->getId()<<"  track filter position: "<<track_filter_x<<" "<<track_filter_y<<" "<<track_filter_z<<"  track rate:"<<distance/time_diff<<"  distance:"<<distance<<"  time_diff: "<<time_diff<<"  track name:"<<track_object_name<<"  unassociated_object_name:"<<unassociated_detection_object_name<<std::endl;
            //                 std::cout<<"track id:"<<t->getId()<<"  track rate:"<<distance/time_diff<<"  distance:"<<distance<<"  time_diff: "<<time_diff<<"  track name:"<<track_object_name<<"  unassociated_object_name:"<<unassociated_detection_object_name<<std::endl;

            //                 if(track_object_name==(unassociated_detection_object_name+"_distract_track")&&(distance/time_diff)<1)
            //                 {
            //                     distract_detection=true;
            //                     break;
            //                 }

            //                if(track_object_name==unassociated_detection_object_name&&(distance/time_diff)>3)
            //                {
            //                    distract_detection=true;
            //                }
            //                else{

            //                }
            //            }
            if(t->getStatus()==TrackObject::NORMAL&& t->isValidated()&&track_object_name==unassociated_detection_object_name)
            {
                double track_filter_x,track_filter_y,track_filter_z;
                t->getState(track_filter_x, track_filter_y, track_filter_z);
                double distance= sqrt(pow((track_filter_x-detection_position[0]),2)+pow((track_filter_y-detection_position[1]),2)+pow((track_filter_z-detection_position[2]),2));
                float time_diff=t->getSecFromLastHighConfidenceDetection(unassociated_detection_time);
                //                std::cout<<"track id:"<<t->getId()<<"  track filter position: "<<track_filter_x<<" "<<track_filter_y<<" "<<track_filter_z<<"  track rate:"<<distance/time_diff<<"  distance:"<<distance<<"  time_diff: "<<time_diff<<"  track name:"<<track_object_name<<"  unassociated_object_name:"<<unassociated_detection_object_name<<std::endl;
                std::cout<<"track id:"<<t->getId()<<"  track rate:"<<distance/time_diff<<"  distance:"<<distance<<"  time_diff: "<<time_diff<<"  track name:"<<track_object_name<<"  unassociated_object_name:"<<unassociated_detection_object_name<<std::endl;

                if((distance/time_diff)<10)
                {
                    distract_detection=false;
                    break;
                }
                else
                    distract_detection=true;
            }
        }


      createNewTrack(*dit, distract_detection);
*/
    }
}

void
TrackerObject::setMinConfidenceForTrackInitialization (double min_confidence)
{
    min_confidence_ = min_confidence;
}

void
TrackerObject::setSecBeforeOld (double sec_before_old)
{
    sec_before_old_ = sec_before_old;
}

void
TrackerObject::setSecBeforeFake (double sec_before_fake)
{
    sec_before_fake_ = sec_before_fake;
}

void
TrackerObject::setSecRemainNew (double sec_remain_new)
{
    sec_remain_new_ = sec_remain_new;
}

void
TrackerObject::setDetectionsToValidate (int detections_to_validate)
{
    detections_to_validate_ = detections_to_validate;
}

void
TrackerObject::setDetectorLikelihood (bool detector_likelihood)
{
    detector_likelihood_ = detector_likelihood;
}

void
TrackerObject::setLikelihoodWeights (double detector_weight, double motion_weight)
{
    likelihood_weights_[0] = detector_weight;
    likelihood_weights_[1] = motion_weight;
}

void
TrackerObject::setVelocityInMotionTerm (bool velocity_in_motion_term, double acceleration_variance, double position_variance)
{
    velocity_in_motion_term_ = velocity_in_motion_term;
    acceleration_variance_ = acceleration_variance;
    position_variance_ = position_variance;

    // Update all existing tracks:
    for(std::list<open_ptrack::tracking::TrackObject*>::iterator it = tracks_.begin(); it != tracks_.end(); it++)
    {
        open_ptrack::tracking::TrackObject* t = *it;
        t->setVelocityInMotionTerm (velocity_in_motion_term_, acceleration_variance_, position_variance_);
    }
}

void
TrackerObject::setAccelerationVariance (double acceleration_variance)
{
    acceleration_variance_ = acceleration_variance;

    // Update all existing tracks:
    for(std::list<open_ptrack::tracking::TrackObject*>::iterator it = tracks_.begin(); it != tracks_.end(); it++)
    {
        open_ptrack::tracking::TrackObject* t = *it;
        t->setAccelerationVariance (acceleration_variance_);
    }
}

void
TrackerObject::setPositionVariance (double position_variance)
{
    position_variance_ = position_variance;

    // Update all existing tracks:
    for(std::list<open_ptrack::tracking::TrackObject*>::iterator it = tracks_.begin(); it != tracks_.end(); it++)
    {
        open_ptrack::tracking::TrackObject* t = *it;
        t->setPositionVariance (position_variance_);
    }
}

void
TrackerObject::setGateDistance (double gate_distance)
{
    gate_distance_ = gate_distance;
}
} /* namespace tracking */
} /* namespace open_ptrack */
