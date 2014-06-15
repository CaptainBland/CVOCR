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




/* Neural Network class - responsible for... being a neural network. Implements evaluation of inputs to the neural network
will probably add backprop to it at some point

This is not really intended to be the 'end' interface - i.e. this requires a facade.*/
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
    
    /*Set the default weights matrix to the neural network
      -- This does no extra work other than setting the weights matrix as is.
      @param network - example: if it has a maximum of 10 neurons per layer, and five layers, the network matrix should be 10x5.
    */
    void setNetwork(cv::Mat network)
    {
        //this is our default weights matrix. Usually will be randomly between 0 and 1 assigned except the input should always be 1
        //these will be adjusted when backprop is implemented. This is very 'todo'
        this->weights = network;
    }
    
    /*Set the activations matrices.
        So this essentially represents synapses. Each layer has a matrix which represents the activations which
        happen at that node from the layer that came before it. I think this should generate a 'feedforward' style network
        
        @param activations - matrix elements are between 0 and 1 depending on how much weight they have to the following synapse
     */
    void setActivations(std::vector<cv::Mat> activations)
    {
        this->activations = activations;
    }
    
    /*eval - evaluate the function the neural network represents with respect to the input matrix.
        @param inputs - a column vector filled with binary inputs which represent our features 
        @return a column vector with numbers between 0 and 1 which represent our output.
    */
    cv::Mat eval(cv::Mat input)
    {
        assert(input.rows == activations[0].rows); //make some vague assertion that the rows on the input are the same as at least the next layer
        cv::Mat rinput = input.t(); 
        cv::Mat activations = 
        //for each layer... (todo: vectorise moar) 
        for(int layer = 1; layer < input.cols; layer++)
        {
            
        }
        //the returned result is whatever the output of the output layer is.
        //so in an AND gate example, we expect the neuron in the second row to always be zero (so only row 1 is meaningful)
        
        
        
        
        return cv::Mat::zeros(2,2, CV_32FC1);
    }
    
    
    
    
    
    
    ~NeuralNetwork()
    {
        //clean up in here
    }
};



