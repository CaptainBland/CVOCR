#include <iostream>
#include "NeuralNet.hpp"
#include <sstream>
NeuralNetwork *makeGateWithWeights(float weightA, float weightB, float weightC)
{
    //let's make an AND gate for the time being (e.g. Keep It Simple, Stupid!)

    //basically have it so layer n, node 2 does not receive input
    cv::Mat thetaHidden = (cv::Mat_<float>(3,3) << weightA, 0, 0, weightB, 0, 0, weightC, 0, 0);
    std::cout<<thetaHidden<<std::endl;
    

    //std::cout<<thetaOut<<std::endl;
    std::vector<cv::Mat> thetaParams;
    thetaParams.push_back(thetaHidden);
    //thetaParams.push_back(thetaOut);
    
    NeuralNetwork *mynet = new NeuralNetwork;
    
    mynet->setWeights(thetaParams);
    
    
    return mynet;
}


//unit tests for gates
bool testAND()
{
 
    std::cout<<"AND test: " << std::endl;   
    bool result = true;
    float b = 1;
    auto gate = makeGateWithWeights(-30, 20, 20);

    //I should come up with a better representation for the bias unit but this will do!
    cv::Mat t = gate->eval((cv::Mat_<float>(3,1) << b,1,1));
    cv::Mat f1 = gate->eval((cv::Mat_<float>(3,1) << b,1,0));
    cv::Mat f2 = gate->eval((cv::Mat_<float>(3,1) << b,0,1));
    cv::Mat f3 = gate->eval((cv::Mat_<float>(3,1) << b,0,0));
    
    std::stringstream matstream;

    matstream<<"AND results: " << t << f1 << f2 << f3; 

    Log(SPAM, matstream.str());

    result = result && t.at<float>(0,0) == 1;
    result = result && f1.at<float>(0,0) != 1;
    result = result && f2.at<float>(0,0) != 1;
    result = result && f3.at<float>(0,0) != 1;
    delete gate;
    return result;
}

//unit tests for gates
bool testOR()
{
 
    std::cout<<"AND test: " << std::endl;   
    bool result = true;
    float b = 1;
    auto gate = makeGateWithWeights(-10, 20, 20);

    //I should come up with a better representation for the bias unit but this will do!
    cv::Mat t1 = gate->eval((cv::Mat_<float>(3,1) << b,1,1));
    cv::Mat t2 = gate->eval((cv::Mat_<float>(3,1) << b,1,0));
    cv::Mat t3 = gate->eval((cv::Mat_<float>(3,1) << b,0,1));
    cv::Mat f1 = gate->eval((cv::Mat_<float>(3,1) << b,0,0));
    
    std::stringstream matstream;

    matstream<<"AND results: " << t1 << t2 << t3 << f1; 

    Log(SPAM, matstream.str());

    result = result && t1.at<float>(0,0) == 1;
    result = result && t2.at<float>(0,0) == 1;
    result = result && t3.at<float>(0,0) == 1;
    result = result && f1.at<float>(0,0) != 1;
    delete gate;
    return result;
}



/*Run NeuralNetwork tests*/
int main(int argc, char ** argv)
{

    Log(DEBUG, std::string("AND success: ") + std::to_string(testAND())); 
    Log(DEBUG, std::string("OR success: ") + std::to_string(testOR()));

}



