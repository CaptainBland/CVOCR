#include <iostream>
#include "NeuralNet.hpp"

NeuralNetwork *hardCodedAnd()
{
    //let's make an AND gate for the time being (e.g. Keep It Simple, Stupid!)
    
    //Activation matrix is 2x3
    
    cv::Mat inputs = cv::Mat::zeros(2,1, CV_32FC1); //let our input matrix be zero, zero
    cv::Mat hidden = cv::Mat::zeros(2,1, CV_32FC1);
    cv::Mat output = cv::Mat::zeros(2,1, CV_32FC1);
    
    
    /* [1 0
        1 0]
    */
    cv::Mat thetaHidden = (cv::Mat_<float>(2,2) << 30, 0, -20, 0);
    std::cout<<thetaHidden<<std::endl;
    
    /*
       [1 0
        0 0]
    */
    cv::Mat thetaOut = (cv::Mat_<float>(2,2) << 1, 0, 0, 0);
    std::cout<<thetaOut<<std::endl;
    std::vector<cv::Mat> thetaParams;
    thetaParams.push_back(thetaHidden);
    thetaParams.push_back(thetaOut);
    
    NeuralNetwork *mynet = new NeuralNetwork;
    
    mynet->setActivations(thetaParams);
    
    
    /*
    [0 1 1
     0 1 1]
    */
    cv::Mat defaultWeights = (cv::Mat_<float>(2,3) << 0, 1, 1, 0, 1, 1);
    std::cout<<defaultWeights<<std::endl;
    mynet->setNetwork(defaultWeights);
    
    return mynet;
}



/*Run NeuralNetwork tests*/
int main(int argc, char ** argv)
{
    NeuralNetwork *hca = hardCodedAnd();
    cv::Mat inputs = (cv::Mat_<float>(2,1) << -1, 0);
    
    std::cout<<hca->eval(inputs);
    
    
	std::cout<<"Todo: all."<<std::endl;
	
	delete hca;
}



