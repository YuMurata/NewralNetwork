#pragma once

#include"Newron.h"

class Layer
{
public:
	vector<Newron> layer;
	int input_num;
	int output_num;

	Layer(const int &input_num, const int &output_num, const Newron::ActFunc &func)
		:input_num(input_num), output_num(output_num)
	{
		this->layer.reserve(output_num);
		for (int i = 0; i < output_num; ++i)
		{
			this->layer.push_back(Newron(input_num, func));
		}
	}

	VectorXd Forward(const VectorXd &input)
	{
		auto ret_size = size(layer);
		VectorXd ret(ret_size);

		vector<double> temp;
		temp.reserve(ret_size);
		for (auto &i : this->layer)
		{
			auto value = i.Forward(input);
			temp.push_back(value);
		}

		ret = Map<VectorXd>(temp.data(), ret_size);

		return ret;
	}

	VectorXd Backward(const VectorXd &deltas)
	{
		VectorXd ret = VectorXd::Zero(this->input_num);

		for (int i = 0; i<this->output_num; ++i)
		{
			ret += this->layer[i].Backward(deltas(i));
		}

		return ret;
	}
};
