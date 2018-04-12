#include "histogram_.h"
#include <cstring>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>

Histogramt::Histogramt(int _dimSize, int _range)
{
    dimSize = _dimSize;
    range = _range;

    rangePerBin = range/dimSize;
    rangePerBinInv = 1./(float)rangePerBin;

    // Set look up tables
    for(int i = 0; i < 256; i++){
        lookUpTable1[i] = (int)(rangePerBinInv*i);
        lookUpTable2[i] = lookUpTable1[i]*dimSize;
        lookUpTable3[i] = lookUpTable2[i]*dimSize;
    }
}

void Histogramt::insertValues(std::vector<unsigned char> & data1, std::vector<unsigned char> & data2, std::vector<unsigned char> & data3, std::vector<double> &weight)
{
    if (data.size() < (unsigned int)dimSize*dimSize*dimSize)
        data.resize(dimSize*dimSize*dimSize);

    bool useWeights = true;
    if (weight.size() != data1.size())
        useWeights = false;

    double sum = 0;
    for (unsigned int i=0; i < data1.size(); ++i){
        int id1 = lookUpTable1[data1[i]];
        int id2 = lookUpTable2[data2[i]];
        int id3 = lookUpTable3[data3[i]];
        int id = id1 + id2 + id3;

        double w = useWeights ? weight[i] : 1;

        data[id] += w;
        sum += w;
    }

    normalize();
}

double Histogramt::computeSimilarity(Histogramt * hist)
{
    double conf = 0;
    for (unsigned int i=0; i < data.size(); ++i) {
        conf += sqrt(data[i]*hist->data[i]);
    }
    return conf;
}

double Histogramt::getValue(int val1, int val2, int val3)
{
    int id1 = lookUpTable1[val1];
    int id2 = lookUpTable2[val2];
    int id3 = lookUpTable3[val3];
    int id = id1 + id2 + id3;
    return data[id];
}

void Histogramt::transformToWeights()
{
    double min = 0;
/*    std::ifstream alfa_file;
    alfa_file.open("param.txt");
    if (alfa_file.is_open()){
        double sum = 0;
        for (unsigned int i=0; i < data.size(); ++i) {
            sum += data[i];
        }
        double alfa;
        alfa_file >> alfa;
        min = (alfa/100.0)*sum;
        alfa_file.close();
    }else*/
        min = getMin();

    transformByWeight(min);
}

void Histogramt::transformByWeight(double min)
{
    for (unsigned int i=0; i < data.size(); ++i){
        if (data[i] > 0){
            data[i] = min/data[i];
            if (data[i] > 1)
                data[i] = 1;
        }else
            data[i] = 1;
    }

}

void Histogramt::multiplyByWeights(Histogramt * hist)
{
    double sum = 0;
    for (unsigned int i=0; i < data.size(); ++i) {
        data[i] *= hist->data[i];
        sum += data[i];
    }

    normalize();
}

void Histogramt::adapt(Histogramt * hist, float height){
    for (unsigned int i=0; i < data.size(); ++i) {
        data[i] *= 1-height;
        data[i] += (hist->data[i]*height);
    }

    normalize();
}


void Histogramt::clear()
{
    for (unsigned int i=0; i < data.size(); ++i)
        data[i] = 0;
}

void Histogramt::normalize()
{
    double sum = 0;
    for (unsigned int i=0; i < data.size(); ++i)
        sum += data[i];
    for (unsigned int i=0; i < data.size(); ++i)
        data[i] /= sum;
}

double Histogramt::getMin()
{
    double min = 1;
    for (unsigned int i=0; i < data.size(); ++i) {
        if (data[i] < min && data[i] != 0)
            min = data[i];
    }
    return min;
}

void Histogramt::addExpHist(double alpha, Histogramt & hist)
{
    double beta = 1-alpha;
    for (unsigned int i=0; i < data.size(); ++i){
        data[i] = beta*data[i] + alpha*hist.data[i];
    }
}

