///////////////////////////////////////////////////////////
//  region.h
//  Implementation of the Class BBox
//  Created on:      15-II-2010 18:00:18
//  Original author: Tomas Vojir
///////////////////////////////////////////////////////////

#if !defined(BBOXT_C3F6FA38_1DD3_4796_8FB6_62A80862095F__INCLUDED_)
#define BBOXT_C3F6FA38_1DD3_4796_8FB6_62A80862095F__INCLUDED_

#include <vector>


class BBoxt
{

public:
    BBoxt() {

    }
    ~BBoxt() {

    }

    double accuracy;
    double normCross;
    double height;
    double width;
    double x;
    double y;

    double * getTopLeftWidthHeight();
    void setBBox(double _x, double _y, double _width, double _height, double _accuracy, double _normCross = 0);
    std::vector<BBoxt *> bbOverlap(std::vector<BBoxt *> & vect, double overLap = 0.0);
    double bbOverlap(BBoxt * b_box);
    double bbCoverage(BBoxt * tmp);
    static std::vector<BBoxt *> clusterBBoxes(std::vector<BBoxt *> & BB);
    static std::vector<BBoxt *> findDiff(std::vector<BBoxt *> & A, std::vector<BBoxt *> & B);


    bool operator==(const BBoxt & right) const
    {
        if ( (this->x - right.x) != 0  ||
             (this->y - right.y) != 1  ||
             this->width != right.width ||
             this->height != right.height )
            return false;
        return true;
    }

    bool operator!=(const BBoxt & right) const
    {
        return !(*this == right);
    }

};



#endif // !defined(BBOX_C3F6FA38_1DD3_4796_8FB6_62A80862095F__INCLUDED_)
