#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <random>
#include <array>
#include <vector>
#include <algorithm>

typedef uint32_t uint32;
typedef uint16_t uint16;
typedef uint8_t uint8;

const size_t c_numInputNeurons = 784;
const size_t c_numHiddenNeurons = 30;
const size_t c_numOutputNeurons = 10;

const size_t c_trainingEpochs = 30;  //TODO: 30
const size_t c_miniBatchSize = 10;
const float c_learningRate = 3.0f;

// ============================================================================================
//                                    MNIST DATA LOADER
// ============================================================================================

inline uint32 EndianSwap (uint32 a)
{
	return (a<<24) | ((a<<8) & 0x00ff0000) |
           ((a>>8) & 0x0000ff00) | (a>>24);
}

// MNIST data and file format description is from http://yann.lecun.com/exdb/mnist/
class CMNISTData
{
public:
	CMNISTData ()
	{
		m_labelData = nullptr;
		m_imageData = nullptr;

		m_imageCount = 0;
		m_labels = nullptr;
		m_pixels = nullptr;
	}

	bool Load (bool training)
	{
		// set the expected image count
		m_imageCount = training ? 60000 : 10000;

		// read labels
		const char* labelsFileName = training ? "train-labels.idx1-ubyte" : "t10k-labels.idx1-ubyte";
		FILE* file = fopen(labelsFileName,"rb");
		if (!file)
		{
			printf("could not open %s for reading.\n", labelsFileName);
			return false;
		}
		fseek(file, 0, SEEK_END);
		long fileSize = ftell(file);
		fseek(file, 0, SEEK_SET);
		m_labelData = new uint8[fileSize];
		fread(m_labelData, fileSize, 1, file);
		fclose(file);

		// read images
		const char* imagesFileName = training ? "train-images.idx3-ubyte" : "t10k-images.idx3-ubyte";
		file = fopen(imagesFileName, "rb");
		if (!file)
		{
			printf("could not open %s for reading.\n", imagesFileName);
			return false;
		}
		fseek(file, 0, SEEK_END);
		fileSize = ftell(file);
		fseek(file, 0, SEEK_SET);
		m_imageData = new uint8[fileSize];
		fread(m_imageData, fileSize, 1, file);
		fclose(file);

		// endian swap label file if needed, just first two uint32's.  The rest is uint8's.
		uint32* data = (uint32*)m_labelData;
		if (data[0] == 0x01080000)
		{
			data[0] = EndianSwap(data[0]);
			data[1] = EndianSwap(data[1]);
		}

		// verify that the label file has the right header
		if (data[0] != 2049 || data[1] != m_imageCount)
		{
			printf("Label data had unexpected header values.\n");
			return false;
		}
		m_labels = (uint8*)&(data[2]);

		// endian swap the image file if needed, just first 4 uint32's. The rest is uint8's.
		data = (uint32*)m_imageData;
		if (data[0] == 0x03080000)
		{
			data[0] = EndianSwap(data[0]);
			data[1] = EndianSwap(data[1]);
			data[2] = EndianSwap(data[2]);
			data[3] = EndianSwap(data[3]);
		}

		// verify that the image file has the right header
		if (data[0] != 2051 || data[1] != m_imageCount || data[2] != 28 || data[3] != 28)
		{
			printf("Label data had unexpected header values.\n");
			return false;
		}
		m_pixels = (uint8*)&(data[4]);

		// convert the pixels from uint8 to float
		m_pixelsFloat.resize(28 * 28 * m_imageCount);
		for (size_t i = 0; i < 28 * 28 * m_imageCount; ++i)
			m_pixelsFloat[i] = float(m_pixels[i]) / 255.0f;

		// success!
		return true;
	}

	~CMNISTData ()
	{
		delete[] m_labelData;
		delete[] m_imageData;
	}

	size_t NumImages () const { return m_imageCount; }

	const float* GetImage (size_t index, uint8& label) const
	{
		label = m_labels[index];
		return &m_pixelsFloat[index * 28 * 28];
	}

private:
	void* m_labelData;
	void* m_imageData;

	size_t m_imageCount;
	uint8* m_labels;
	uint8* m_pixels;

	std::vector<float> m_pixelsFloat;
};

// ============================================================================================
//                                    NEURAL NETWORK
// ============================================================================================

template <size_t INPUTS, size_t HIDDEN_NEURONS, size_t OUTPUT_NEURONS>
class CNeuralNetwork
{
public:
	CNeuralNetwork ()
	{
		// initialize weights and biases to a gaussian distribution random number with mean 0, stddev 1.0
		std::random_device rd;
		std::mt19937 e2(rd());
		std::normal_distribution<float> dist(0, 1);

		for (float& f : m_hiddenLayerBiases)
			f = dist(e2);

		for (float& f : m_outputLayerBiases)
			f = dist(e2);

		for (float& f : m_hiddenLayerWeights)
			f = dist(e2);

		for (float& f : m_outputLayerWeights)
			f = dist(e2);
	}

	void Train (const CMNISTData& trainingData, size_t miniBatchSize, float learningRate, float& avgError)
	{
		/*
		// TODO: temp!
		{
			float error = 0.0f;
			float input[2] = { 0.05f, 0.1f };
			uint8 imageLabel = 0;
			uint8 labelDetected = ForwardPass(input, imageLabel, error);
			avgError += error;

			// run the backward pass to get derivatives of the cost function
			BackwardPass(input, imageLabel);

			int ijkl = 0;
		}
		*/


		// shuffle the order of the training data for our mini batches
		if (m_trainingOrder.size() != trainingData.NumImages())
		{
			m_trainingOrder.resize(trainingData.NumImages());
			size_t index = 0;
			for (size_t& v : m_trainingOrder)
			{
				v = index;
				++index;
			}
		}
		static std::random_device rd;
		static std::mt19937 e2(rd());
		std::shuffle(m_trainingOrder.begin(), m_trainingOrder.end(), e2);

		// initialize average error to 0
		avgError = 0.0f;

		// process all minibatches until we are out of training examples
		size_t trainingIndex = 0;
		while (trainingIndex < trainingData.NumImages())
		{
			// Clear out minibatch derivatives.  We sum them up and then divide at the end of the minimatch
			std::fill(m_miniBatchHiddenLayerBiasesDeltaCost.begin(), m_miniBatchHiddenLayerBiasesDeltaCost.end(), 0.0f);
			std::fill(m_miniBatchOutputLayerBiasesDeltaCost.begin(), m_miniBatchOutputLayerBiasesDeltaCost.end(), 0.0f);
			std::fill(m_miniBatchHiddenLayerWeightsDeltaCost.begin(), m_miniBatchHiddenLayerWeightsDeltaCost.end(), 0.0f);
			std::fill(m_miniBatchOutputLayerWeightsDeltaCost.begin(), m_miniBatchOutputLayerWeightsDeltaCost.end(), 0.0f);

			// process the minibatch
			size_t miniBatchIndex = 0;
			while (miniBatchIndex < miniBatchSize && trainingIndex < trainingData.NumImages())
			{
				// get the training item
				uint8 imageLabel = 0;
				const float* pixels = trainingData.GetImage(m_trainingOrder[trainingIndex], imageLabel);

				// run the forward pass of the network
				float error = 0.0f;
				uint8 labelDetected = ForwardPass(pixels, imageLabel, error);
				avgError += error;

				// run the backward pass to get derivatives of the cost function
				BackwardPass(pixels, imageLabel);

				// add the current derivatives into the minibatch derivative arrays so we can average them at the end of the minibatch via division.
				for (size_t i = 0; i < m_hiddenLayerBiasesDeltaCost.size(); ++i)
					m_miniBatchHiddenLayerBiasesDeltaCost[i] += m_hiddenLayerBiasesDeltaCost[i];
				for (size_t i = 0; i < m_outputLayerBiasesDeltaCost.size(); ++i)
					m_miniBatchOutputLayerBiasesDeltaCost[i] += m_outputLayerBiasesDeltaCost[i];
				for (size_t i = 0; i < m_hiddenLayerWeightsDeltaCost.size(); ++i)
					m_miniBatchHiddenLayerWeightsDeltaCost[i] += m_hiddenLayerWeightsDeltaCost[i];
				for (size_t i = 0; i < m_outputLayerWeightsDeltaCost.size(); ++i)
					m_miniBatchOutputLayerWeightsDeltaCost[i] += m_outputLayerWeightsDeltaCost[i];

				// note that we've added another item to the minibatch, and that we've consumed another training example
				++trainingIndex;
				++miniBatchIndex;
			}

			// divide minibatch derivatives by how many items were in the minibatch, to get the average of the derivatives.
			// instead of the commented code below, we'll do it implicitly by dividing the learning rate by miniBatchIndex.
			/*
			for (float& f : m_miniBatchHiddenLayerBiasesDeltaCost)
				f /= float(miniBatchIndex);
			for (float& f : m_miniBatchOutputLayerBiasesDeltaCost)
				f /= float(miniBatchIndex);
			for (float& f : m_miniBatchHiddenLayerWeightsDeltaCost)
				f /= float(miniBatchIndex);
			for (float& f : m_miniBatchOutputLayerWeightsDeltaCost)
				f /= float(miniBatchIndex);
			*/

			// apply training to biases and weights
			float miniBatchLearningRate = learningRate / float(miniBatchIndex);

			for (size_t i = 0; i < m_hiddenLayerBiases.size(); ++i)
				m_hiddenLayerBiases[i] -= m_miniBatchHiddenLayerBiasesDeltaCost[i] * miniBatchLearningRate;
			for (size_t i = 0; i < m_outputLayerBiases.size(); ++i)
				m_outputLayerBiases[i] -= m_miniBatchOutputLayerBiasesDeltaCost[i] * miniBatchLearningRate;
			for (size_t i = 0; i < m_hiddenLayerWeights.size(); ++i)
				m_hiddenLayerWeights[i] -= m_miniBatchHiddenLayerWeightsDeltaCost[i] * miniBatchLearningRate;
			for (size_t i = 0; i < m_outputLayerWeights.size(); ++i)
				m_outputLayerWeights[i] -= m_miniBatchOutputLayerWeightsDeltaCost[i] * miniBatchLearningRate;
		}

		avgError /= float(trainingData.NumImages());
	}

	// This function evaluates the network for the given input pixels and returns the label it thinks it is from 0-9
	uint8 ForwardPass (const float* pixels, uint8 correctLabel, float& error)
	{
		// first do hidden layer
		for (size_t neuronIndex = 0; neuronIndex < HIDDEN_NEURONS; ++neuronIndex)
		{
			float Z = m_hiddenLayerBiases[neuronIndex];

			for (size_t inputIndex = 0; inputIndex < INPUTS; ++inputIndex)
				Z += pixels[inputIndex] * m_hiddenLayerWeights[HiddenLayerWeightIndex(inputIndex, neuronIndex)];

			m_hiddenLayerOutputs[neuronIndex] = 1.0f / (1.0f + std::exp(-Z));

			// TODO: temp
			int ijkl = 0;
		}

		// then do output layer
		for (size_t neuronIndex = 0; neuronIndex < OUTPUT_NEURONS; ++neuronIndex)
		{
			float Z = m_outputLayerBiases[neuronIndex];

			for (size_t inputIndex = 0; inputIndex < HIDDEN_NEURONS; ++inputIndex)
				Z += m_hiddenLayerOutputs[inputIndex] * m_outputLayerWeights[OutputLayerWeightIndex(inputIndex, neuronIndex)];

			m_outputLayerOutputs[neuronIndex] = 1.0f / (1.0f + std::exp(-Z));

			// TODO: temp
			int ijkl = 0;
		}

		// calculate error.
		// this is the magnitude of the vector that is Desired - Actual
		{
			error = 0.0f;
			for (size_t neuronIndex = 0; neuronIndex < OUTPUT_NEURONS; ++neuronIndex)
			{
				// TODO: temp!
				float desiredOutput = (correctLabel == neuronIndex) ? 1.0f : 0.0f;
				//float desiredOutput = (neuronIndex == 0) ? 0.01f : 0.99f;
				float diff = (desiredOutput - m_outputLayerOutputs[neuronIndex]);
				error += diff * diff;
			}
			error = std::sqrt(error);
		}

		// find the maximum value of the output layer and return that index as the label
		float maxOutput = m_outputLayerOutputs[0];
		uint8 maxLabel = 0;
		for (uint8 neuronIndex = 1; neuronIndex < OUTPUT_NEURONS; ++neuronIndex)
		{
			if (m_outputLayerOutputs[neuronIndex] > maxOutput)
			{
				maxOutput = m_outputLayerOutputs[neuronIndex];
				maxLabel = neuronIndex;
			}
		}
		return maxLabel;
	}

// TODO: temp!
//private:

	static size_t HiddenLayerWeightIndex (size_t inputIndex, size_t hiddenLayerNeuronIndex)
	{
		return hiddenLayerNeuronIndex * INPUTS + inputIndex;
	}

	static size_t OutputLayerWeightIndex (size_t hiddenLayerNeuronIndex, size_t outputLayerNeuronIndex)
	{
		return outputLayerNeuronIndex * HIDDEN_NEURONS + hiddenLayerNeuronIndex;
	}

	// this function uses the neuron output values from the forward pass to backpropagate the error
	// of the network to calculate the gradient needed for training.  It figures out what the error
	// is by comparing the label it came up with to the label it should have come up with (correctLabel).
	void BackwardPass (const float* pixels, uint8 correctLabel)
	{
		// since we are going backwards, do the output layer first
		for (size_t neuronIndex = 0; neuronIndex < OUTPUT_NEURONS; ++neuronIndex)
		{
			// calculate deltaCost/deltaBias for each output neuron.
			// This is also the error for the neuron, and is the same value as deltaCost/deltaZ.
			//
			// deltaCost/deltaZ = deltaCost/deltaO * deltaO/deltaZ
			//
			// deltaCost/deltaO = O - desiredOutput
			// deltaO/deltaZ = O * (1 - O)
			//
			// TODO: temp!
			float desiredOutput = (correctLabel == neuronIndex) ? 1.0f : 0.0f;
			//float desiredOutput = (neuronIndex == 0) ? 0.01f : 0.99f;

			float deltaCost_deltaO = m_outputLayerOutputs[neuronIndex] - desiredOutput;
			float deltaO_deltaZ = m_outputLayerOutputs[neuronIndex] * (1.0f - m_outputLayerOutputs[neuronIndex]);

			m_outputLayerBiasesDeltaCost[neuronIndex] = deltaCost_deltaO * deltaO_deltaZ;

			// calculate deltaCost/deltaWeight for each weight going into the neuron
			//
			// deltaCost/deltaWeight = deltaCost/deltaZ * deltaCost/deltaWeight
			// deltaCost/deltaWeight = deltaCost/deltaBias * input
			//
			for (size_t inputIndex = 0; inputIndex < HIDDEN_NEURONS; ++inputIndex)
				m_outputLayerWeightsDeltaCost[OutputLayerWeightIndex(inputIndex, neuronIndex)] = m_outputLayerBiasesDeltaCost[neuronIndex] * m_hiddenLayerOutputs[inputIndex];
		}

		// then do the hidden layer
		for (size_t neuronIndex = 0; neuronIndex < HIDDEN_NEURONS; ++neuronIndex)
		{
			// calculate deltaCost/deltaBias for each hidden neuron.
			// This is also the error for the neuron, and is the same value as deltaCost/deltaZ.
			//
			// deltaCost/deltaO =
			//   Sum for each output of this neuron:
			//     deltaCost/deltaDestinationZ * deltaDestinationZ/deltaSourceO
			//
			// deltaCost/deltaDestinationZ is already calculated and lives in m_outputLayerBiasesDeltaCost[destinationNeuronIndex].
			// deltaTargetZ/deltaSourceO is the value of the weight connecting the source and target neuron.
			//
			// deltaCost/deltaZ = deltaCost/deltaO * deltaO/deltaZ
			// deltaO/deltaZ = O * (1 - O)
			//
			float deltaCost_deltaO = 0.0f;
			for (size_t destinationNeuronIndex = 0; destinationNeuronIndex < OUTPUT_NEURONS; ++destinationNeuronIndex)
				deltaCost_deltaO += m_outputLayerBiasesDeltaCost[destinationNeuronIndex] * m_outputLayerWeights[OutputLayerWeightIndex(neuronIndex, destinationNeuronIndex)];
			float deltaO_deltaZ = m_hiddenLayerOutputs[neuronIndex] * (1.0f - m_hiddenLayerOutputs[neuronIndex]);
			m_hiddenLayerBiasesDeltaCost[neuronIndex] = deltaCost_deltaO * deltaO_deltaZ;

			// calculate deltaCost/deltaWeight for each weight going into the neuron
			//
			// deltaCost/deltaWeight = deltaCost/deltaZ * deltaCost/deltaWeight
			// deltaCost/deltaWeight = deltaCost/deltaBias * input
			//
			for (size_t inputIndex = 0; inputIndex < INPUTS; ++inputIndex)
				m_hiddenLayerWeightsDeltaCost[HiddenLayerWeightIndex(inputIndex, neuronIndex)] = m_hiddenLayerBiasesDeltaCost[neuronIndex] * pixels[inputIndex];
		}
	}

// TODO: temp!
//private:

	// biases and weights
	std::array<float, HIDDEN_NEURONS>					m_hiddenLayerBiases;
	std::array<float, OUTPUT_NEURONS>					m_outputLayerBiases;

	std::array<float, INPUTS * HIDDEN_NEURONS>			m_hiddenLayerWeights;
	std::array<float, HIDDEN_NEURONS * OUTPUT_NEURONS>	m_outputLayerWeights;

	// neuron activation values aka "O" values
	std::array<float, HIDDEN_NEURONS>					m_hiddenLayerOutputs;
	std::array<float, OUTPUT_NEURONS>					m_outputLayerOutputs;

	// derivatives of biases and weights for a single training example
	std::array<float, HIDDEN_NEURONS>					m_hiddenLayerBiasesDeltaCost;
	std::array<float, OUTPUT_NEURONS>					m_outputLayerBiasesDeltaCost;

	std::array<float, INPUTS * HIDDEN_NEURONS>			m_hiddenLayerWeightsDeltaCost;
	std::array<float, HIDDEN_NEURONS * OUTPUT_NEURONS>	m_outputLayerWeightsDeltaCost;

	// derivatives of biases and weights for the minibatch. Average of all items in minibatch.
	std::array<float, HIDDEN_NEURONS>					m_miniBatchHiddenLayerBiasesDeltaCost;
	std::array<float, OUTPUT_NEURONS>					m_miniBatchOutputLayerBiasesDeltaCost;

	std::array<float, INPUTS * HIDDEN_NEURONS>			m_miniBatchHiddenLayerWeightsDeltaCost;
	std::array<float, HIDDEN_NEURONS * OUTPUT_NEURONS>	m_miniBatchOutputLayerWeightsDeltaCost;

	// used for minibatch generation
	std::vector<size_t>									m_trainingOrder;
};

// ============================================================================================
//                                   DRIVER PROGRAM
// ============================================================================================

// training and test data
CMNISTData g_trainingData;
CMNISTData g_testData;

// neural network
CNeuralNetwork<c_numInputNeurons, c_numHiddenNeurons, c_numOutputNeurons> g_neuralNetwork;

float GetDataAccuracy (const CMNISTData& data)
{
	size_t correctItems = 0;
	for (size_t i = 0, c = data.NumImages(); i < c; ++i)
	{
		uint8 label;
		const float* pixels = data.GetImage(i, label);
		float error = 0.0f;
		uint8 detectedLabel = g_neuralNetwork.ForwardPass(pixels, label, error);

		if (detectedLabel == label)
			++correctItems;
	}
	return float(correctItems) / float(data.NumImages());
}

int main (int argc, char** argv)
{
	/*
	// TODO: temp!
	CNeuralNetwork<2, 2, 2> nn;
	nn.m_hiddenLayerWeights[0] = 0.15f;
	nn.m_hiddenLayerWeights[1] = 0.2f;
	nn.m_hiddenLayerWeights[2] = 0.25f;
	nn.m_hiddenLayerWeights[3] = 0.3f;

	nn.m_hiddenLayerBiases[0] = 0.35f;
	nn.m_hiddenLayerBiases[1] = 0.35f;

	nn.m_outputLayerWeights[0] = 0.4f;
	nn.m_outputLayerWeights[1] = 0.45f;
	nn.m_outputLayerWeights[2] = 0.5f;
	nn.m_outputLayerWeights[3] = 0.55f;
	
	nn.m_outputLayerBiases[0] = 0.6f;
	nn.m_outputLayerBiases[1] = 0.6f;

	float avgErrorZ = 0.0f;
	nn.Train(g_trainingData, c_miniBatchSize, c_learningRate, avgErrorZ);
	*/



	// load the MNIST data if we can
	if (!g_trainingData.Load(true) || !g_testData.Load(false))
	{
		printf("Could not load mnist data, aborting!\n");
		return 1;
	}

	// train the network
	for (size_t epoch = 0; epoch < c_trainingEpochs; ++epoch)
	{
		printf("epoch %zu / %zu\n", epoch+1, c_trainingEpochs);
		float avgError = 0.0f;
		printf("  Training Data Accuracy: %0.2f%%\n", 100.0f*GetDataAccuracy(g_trainingData));
		printf("  Test Data Accuracy: %0.2f%%\n", 100.0f*GetDataAccuracy(g_testData));
		g_neuralNetwork.Train(g_trainingData, c_miniBatchSize, c_learningRate, avgError);
	}
	printf("\nFinal Training Data Accuracy: %0.2f%%\n", 100.0f*GetDataAccuracy(g_trainingData));
	printf("Final Test Data Accuracy: %0.2f%%\n", 100.0f*GetDataAccuracy(g_testData));

	// TODO: check error of test data

	// TODO: put this code somewhere and comment it out or something to let people verify that the data is loaded correctly
	uint8 label;
	const float* pixels = g_trainingData.GetImage(2, label);
	printf("showing a %i\n", label);
	for (int iy = 0; iy < 28; ++iy)
	{
		for (int ix = 0; ix < 28; ++ix)
		{
			if (*pixels < 0.125)
				printf(" ");
			else
				printf("+");
			++pixels;
		}
		printf("\n");
	}

	system("pause");
	return 0;
}

/*

TODO:
? should we figure out a way to let people give their own input to test network?
 * maybe make an html5 web app to draw numbers and let it guess what it is?
 * should sample the bounding box of the number
* forward pass may not need to calculate error, check into it.
 * same with Train()!
? should we calculate derivatives of input and make static that looks like a number or make a number that doesn't look like one to the machine?
* profile and optimize a little.
* show how long it took to run the process.
* show error rate at end. maybe an option to show it for each epoch?
* CSV of error at end of each epoch to show learning rate. Maybe also show error of training data?
* write out json file of weights/biases for use by html demo!

Blog Notes:
* porting network.py from this page: http://neuralnetworksanddeeplearning.com/chap1.html
* describe the features of the network
* explain any new ideas that aren't covered in last blog post.
* put code and link to data, but also link to github repo
* not multithreaded / SIMD etc.  meant to be readable and usable for experimentation.
* link to this in recipe: http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf
 * exponential linear unit.
* put mnist.zip file up on blog and link to it

Network Description:
* neuron layers: [784, 30, 10]
* 30 training epochs (all minibatches)
* mini batch size of 10
* learning rate of 3.0  (!!)
? not sure if this is the final network description of network.py or not
 * it isn't. look at the other params tried. maybe mention them in the post?
 * 100 hidden neurons could do better, but not reliably. was more wild results.

NEXT PROJECTS:
* port network2.py and make a blog post. from http://neuralnetworksanddeeplearning.com/chap3.html
* do convolutional version
* do a recurrent neural network thing

*/