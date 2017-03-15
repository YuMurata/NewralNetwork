#pragma once

class Newron
{
public:
	using ActFunc = function<double(const double &)>;
	VectorXd input;
	int input_num;
	double output;

	VectorXd weight;
	double bias;

	ActFunc func;

	Newron(const int &input_num, const ActFunc &func)
		:input_num(input_num), func(func)
	{
		this->bias = ((rand() % 101) - 50)*0.1;
		this->weight = VectorXd::Random(input_num)*0.1;
		auto watch = this->weight.data();
	}

	double Forward(const VectorXd &input)
	{
		this->input = input;
		auto Dot = input.dot(this->weight);
		auto ret = this->func(Dot+this->bias);
		this->output = ret;
		
		return ret;
	}

	VectorXd Backward(const double &delta)
	{
		const auto nw = 1e-2;
		const auto nb = 1e-2;
		VectorXd diff(this->input_num);

		for (int i = 0; i < this->input_num; ++i)
		{
			diff(i) = MathPlus::Differential(this->input(i), func);
		}

		VectorXd dw = nw*delta*diff*this->output;
		double db = nb*delta;
		auto watch = dw.data();
		this->weight -= dw;
		this->bias -= db;
		auto ret = delta*this->weight;

		return ret;
	}
};