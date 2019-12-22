#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <vector>
#include <iostream>

namespace sm
{
	using namespace Eigen;
	using namespace std;

	class SM
	{
	public:
		SM()
		{
			const double eta = 1e-1;
			vector<pair<pair<double, double>, double>> samples{ {{0, 0}, 0}, {{0, 1}, 1}, {{1, 0}, 1}, {{1, 1}, 0} };
			
			MatrixXd w1 = MatrixXd::Random(16, 2);
			VectorXd b1 = VectorXd::Random(16);
			MatrixXd w2 = MatrixXd::Random(1, 16);
			VectorXd b2 = VectorXd::Random(1);

			for (int epoch = 0; epoch < 1000; epoch++)
			{
				vector<pair<pair<double, double>, double>> shuffledSamples = samples;
				random_shuffle(shuffledSamples.begin(), shuffledSamples.end());

				for (int index = 0; index < 4; index++)
				{
					const pair<double, double>& input = shuffledSamples[index].first;
					const double o = shuffledSamples[index].second;
					Vector2d i(input.first, input.second);
					VectorXd l1 = w1 * i + b1;
					VectorXd o1 = l1.cwiseMax(0.01 * l1);
					VectorXd l2 = w2 * o1 + b2;
					VectorXd o2 = l2.cwiseMax(0.02 * l2);
					VectorXd g2 = o2.array() - o;
					g2 = g2.array() * (o2.array() > 0).select(ArrayXd::Constant(o2.size(), 1.0), ArrayXd::Constant(o2.size(), 0.01));
					w2 -= g2 * o1.transpose() * eta;
					b2 -= g2 * eta;
					VectorXd g1 = w2.transpose() * g2;
					g1 = g1.array() * (o1.array() > 0).select(ArrayXd::Constant(o1.size(), 1.0), ArrayXd::Constant(o1.size(), 0.01));
					w1 -= g1 * i.transpose() * eta;
					b1 -= g1 * eta;
					std::cout << input.first << ", " << input.second << ": " << o2 << std::endl;
				}
			}
		}

		virtual ~SM()
		{

		}

	private:

	};
}