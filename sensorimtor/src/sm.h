#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <functional>
#include <vector>
#include <string>
#include <cstdint>
#include <limits>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include <Python.h>

namespace sm
{
	using namespace Eigen;
	using namespace std;

	class SM
	{
	public:
		vector<pair<VectorXd, VectorXd>> GetSamples(const string& imagesPath, const string& labelsPath, size_t sampleCount, size_t inputWidth, size_t inputHeight)
		{
			ifstream images(imagesPath, ifstream::binary);
			ifstream labels(labelsPath, ifstream::binary);

			if (!images || !labels)
			{
				throw runtime_error("");
			}

			images.seekg(16, ifstream::beg);
			labels.seekg(8, ifstream::beg);

			const size_t inputSize = inputWidth * inputHeight;
			vector<uint8_t> imagesBuffer(sampleCount * inputSize), labelsBuffer(sampleCount);
			images.read(reinterpret_cast<char*>(&imagesBuffer[0]), imagesBuffer.size());
			labels.read(reinterpret_cast<char*>(&labelsBuffer[0]), labelsBuffer.size());

			images.close();
			labels.close();

			vector<pair<VectorXd, VectorXd>> samples(sampleCount);

			for (size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++)
			{
				pair<VectorXd, VectorXd>& sample = samples[sampleIndex];
				sample.first = VectorXd(inputSize);
				sample.second = VectorXd::Constant(10, 0.0);
				copy(imagesBuffer.begin() + sampleIndex * inputSize, imagesBuffer.begin() + (sampleIndex + 1) * inputSize, sample.first.data());
				sample.first /= 255.0;
				sample.second[labelsBuffer[sampleIndex]] = 1.0;
			}

			return samples;
		}

		SM()
		{
			/*MatrixXi m = MatrixXi::Random(28, 28);
			m -= m / 2 * 2;
			int x = rand() % (28 - 7), y = rand() % (28 - 7);
			int offset = x * 28 + y;
			MatrixXi map = Map<MatrixXi, 0, Stride<28, 1>>(m.data() + offset, 7, 7);
			cout << m << endl << x << ' ' << y << ' ' << offset << endl << map << endl;
			system("pause");*/

			wchar_t arg[] = L"";
			wchar_t* argv[]{ arg };
			Py_Initialize();
			PySys_SetArgv(1, argv);

			PyRun_SimpleString("import matplotlib.pyplot as plt;import random");

			const double eta = 1e-3;

			vector<pair<VectorXd, VectorXd>> samples = GetSamples("thirdparty/mnist/train-images.idx3-ubyte", "thirdparty/mnist/train-labels.idx1-ubyte", 60000, 28, 28);
			vector<pair<VectorXd, VectorXd>> testSamples = GetSamples("thirdparty/mnist/t10k-images.idx3-ubyte", "thirdparty/mnist/t10k-labels.idx1-ubyte", 10000, 28, 28);

			MatrixXd w1 = MatrixXd::Random(64, 28 * 28);// 7 * 7 + 2);
			VectorXd b1 = VectorXd::Random(64);
			MatrixXd w2 = MatrixXd::Random(10, 64);
			VectorXd b2 = VectorXd::Random(10);
			constexpr double epsilon = numeric_limits<double>().epsilon();
			w1 = sqrt(-2.0 * log((w1.array() * 0.5 + 0.5).max(epsilon))) / w1.rows();
			b1 = sqrt(-2.0 * log((b1.array() * 0.5 + 0.5).max(epsilon))) / b1.rows();
			w2 = sqrt(-2.0 * log((w2.array() * 0.5 + 0.5).max(epsilon))) / w2.rows();
			b2 = sqrt(-2.0 * log((b2.array() * 0.5 + 0.5).max(epsilon))) / b1.rows();

			double prevLoss = 0.0;

			for (int epoch = 0; epoch < 1000; epoch++)
			{
				vector<pair<VectorXd, VectorXd>> shuffledSamples(samples.begin(), samples.end());
				random_shuffle(shuffledSamples.begin(), shuffledSamples.end());
				double loss = 0.0;

				//for (pair<VectorXd, VectorXd>& sample : shuffledSamples)
				//{
				//	int x = rand() % (28 - 7), y = rand() % (28 - 7);
				//	//int x = (28 - 7) / 2, y = (28 - 7) / 2;
				//	int offset = x * 28 + y;
				//	MatrixXd tmp = Map<MatrixXd, 0, Stride<28, 1>>(sample.first.data() + offset, 7, 7);
				//	sample.first = VectorXd(7 * 7 + 2);
				//	sample.first << x, y, Map<VectorXd>(tmp.data(), 7 * 7);
				//}

				for(size_t sampleIndex = 0; sampleIndex < shuffledSamples.size(); sampleIndex++)
				{
					const pair<VectorXd, VectorXd>& sample = shuffledSamples[sampleIndex];
					const VectorXd& i = sample.first;
					const VectorXd& o = sample.second;
					VectorXd l1 = w1 * i + b1;
					VectorXd o1 = l1.cwiseMax(0.01 * l1);
					VectorXd l2 = w2 * o1 + b2;
					VectorXd o2 = l2.cwiseMax(0.02 * l2);
					VectorXd err = o2 - o;
					VectorXd g2 = err.array() * (o2.array() > 0).select(ArrayXd::Constant(o2.size(), 1.0), ArrayXd::Constant(o2.size(), 0.01));
					VectorXd g1 = (w2.transpose() * g2).array() * (o1.array() > 0).select(ArrayXd::Constant(o1.size(), 1.0), ArrayXd::Constant(o1.size(), 0.01));
					w2 -= eta * g2 * o1.transpose();
					b2 -= eta * g2;
					w1 -= eta * g1 * i.transpose();
					b1 -= eta * g1;
					loss += err.transpose() * err;

					if (sampleIndex == shuffledSamples.size() - 1)
					{
						int prediction, answer;
						o2.maxCoeff(&prediction);
						o.maxCoeff(&answer);
						cout << prediction << ' ' << answer << endl;
					}
				}

				/*for (size_t sampleIndex = 0; sampleIndex < testSamples.size(); sampleIndex++)
				{
					const pair<VectorXd, VectorXd>& sample = testSamples[sampleIndex];
					const VectorXd& i = sample.first;
					const VectorXd& o = sample.second;
					VectorXd l1 = w1 * i + b1;
					VectorXd o1 = l1.cwiseMax(0.01 * l1);
					VectorXd l2 = w2 * o1 + b2;
					VectorXd o2 = l2.cwiseMax(0.02 * l2);
					int prediction, answer;
					o2.maxCoeff(&prediction);
					o.maxCoeff(&answer);
					loss += prediction == answer ? 1.0 : 0.0;

					if (sampleIndex == testSamples.size() - 1)
					{
						cout << prediction << ' ' << answer << endl;
					}
				}*/

				loss = 0.5 * loss / shuffledSamples.size();
				std::cout << loss << std::endl;

				if (epoch > 0)
				{
					PyRun_SimpleString(("plt.plot([" + to_string(epoch) + ", " + to_string(epoch + 1) + "], [" + to_string(prevLoss) + ", " + to_string(loss) + "], 'b-');plt.pause(0.001)").c_str());
				}

				prevLoss = loss;
			}
		}

		virtual ~SM()
		{

		}

	private:

	};
}