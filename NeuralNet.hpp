/*NeuralNet.hpp
Copyright CaptainBland (2014)*/

#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <iostream>
#include "logger.hpp"
class NeuralNetwork;



/*Our activation function. Currently implements the logistic function*/
template <typename T>
T activation(T t)
{
    //with 1+t for bias
    //return 1/(1+exp(t));
    return (tanh(t));
}

/**Tuple of training examples and expected outputs*/
struct TrainingData
{
    cv::Mat trainingData;
    cv::Mat expectedOutputs;
};


/* Neural Network class - responsible for... being a neural network. Implements evaluation of inputs to the neural network
will probably add backprop to it at some point

This is not really intended to be the 'end' interface - i.e. this requires a facade.

I think this might be limited to feed-forward neural networks. May have to look into other approaches.*/
class NeuralNetwork
{
    private:
    std::vector<cv::Mat> weights;

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
    
 
    
    /** Set the activations matrices.
        So this essentially represents synapses. Each layer has a matrix which represents the activations which
        happen at that node from the layer that came before it. I think this should generate a 'feedforward' style network
        
        @param weights - matrix elements are between 0 and 1 depending on how much weight they have to the following synapse
     */
    void setWeights(std::vector<cv::Mat> weights)
    {
        this->weights = weights;
    }
    
    /**Solve layer - implements (g(z(i))
        Inputs are a row vector
        Weights are a column vector
    */
    cv::Mat solveLayer(cv::Mat inputs /*x*/,
                       cv::Mat weights /*theta*/)
    {
        Log(SPAM, "solveLayer");
        cv::Mat a(inputs.size(), inputs.type());
        a = a.t();//transpose a - return a row vector
        std::cout<<"WEIGHTS: " << weights << "\n";

        std::cout<<"INPUTS: " << inputs << "\n";
        
        for(int i = 0; i < weights.cols; i++)
        {
            //inputs: 1XN, weights(i): NX1 - output: 1X1
            cv::Mat t_zMat = inputs * weights.col(i);
            
            
            float z = t_zMat.at<float>(0,0);
            float temp_a = activation(z);
            
            a.at<float>(i, 0) = temp_a;
        }
        Log(SPAM, "solved layer");
        return a;
    }    
    
    /*eval - evaluate the function the neural network represents with respect to the input matrix.
        @param inputs - a column vector filled with binary inputs which represent our features 
        @return a column vector with numbers between 0 and 1 which represent our output.
    */
    
    
    cv::Mat eval(cv::Mat input)
    {

        check(weights.size() != 0);
        assert(input.rows == weights[0].rows); //make some vague assertion that the rows on the input are the same as at least the next layer
        Log(SPAM, "After assertion");
        
        
        cv::Mat rinput = input.t(); 
        Log(SPAM, "Transpose");
        //for each layer... (todo: vectorise moar) 
        for(size_t layer = 0; layer < weights.size(); layer++)
        {
           
            cv::Mat t_activations = solveLayer(rinput, weights[layer]);
            //forward propagate
            rinput = t_activations.t(); //codebase soon to become missy elliot reference
            
        }
        //the returned result is whatever the output of the output layer is.
        //so in an AND gate example, we expect the neuron in the second row to always be zero (so only row 1 is meaningful)
        
        //at the end, we hope that the result of the forward propagation should now be held in rinput 
        
        
        return rinput;
    }
    
    
    /**Implements the backpropagation learning algorithm
        @param training data - our list of training examples
        @param error - the target error.
        
    */
    void train(const TrainingData &data, float error=0.1, float rate=1)
    {
        //todo: actually implement backprop
    }
    
    
    
    ~NeuralNetwork()
    {
        //clean up in here
    }
};


#endif
