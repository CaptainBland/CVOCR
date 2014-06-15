#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <iostream>

class NeuralNetwork;


template <typename T>
T activation(T t)
{
	return 1/(1+exp(t));
}

std::vector<cv::Mat> hardCodedAnd()
{
	//let's make an AND gate for the time being (e.g. Keep It Simple, Stupid!)
	
	//Activation matrix is 2x3
	
	cv::Mat inputs = cv::Mat.zeros(2,1, CV_32FC1); //let our input matrix be zero, zero
	cv::Mat hidden = cv::Mat.zeros(2,1, CV_32FC1);
	cv::Mat output = cv::Mat.zeros(2,1, CV_32FC1);
	
	cv::Mat thetaHidden = cv::Mat(Mat_<float>(2,2) << 1, 1, 0, 0);
	cv::Mat thetaOut = cv::Mat(Mat_<float>(2,2) << 1, 0, 0, 0);
	
	std::vector<cv::Mat> thetaParams;
	thetaParams.push_back(thetaHidden, thetaOut);
}



class NeuralNetwork
{
	private:
		std::vector<cv::Mat> activations;
		cv::Mat weights;
		size_t matrixheight;
	public:
	
	NeuralNetwork()
	{
		//allocate stuff in here
	}
	
	NeuralNetwork (NeuralNetwork& net)
	{
		//copy constructor
	}
	
	
	void setNetwork(cv::Mat network)
	{
		//this is our default weights matrix. Usually will be randomly between 0 and 1 assigned except the input should always be 1
		//these will be adjusted when backprop is implemented. This is very 'todo'
		this->weights = network;
	}
	
	void setActivations(std::vector<cv::Mat> activations)
	{
		this->activations = activations;
	}
	
	cv::Mat eval(cv::Mat input)
	{
		assert(input.rows == activations[0].rows); //make some vague assertion that the rows on the input are the same as at least the next layer
		
		
		//the returned result is whatever the output of the output layer is.
		//so in an AND gate example, we expect the neuron in the second row to always be zero (so only row 1 is meaningful)
		
		
		
		
		return cv::Mat::zeros(2,2, CV_32FC1);
	}
	
	
	
	
	
	
	~NeuralNetwork()
	{
		//clean up in here
	}
};
