#pragma once

class Newron
{
public:
	using ActFunc = function<double(const double &)>;
	double input;
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
		auto dot = input.dot(this->weight);
		this->input = dot+this->bias;
		this->output = this->func(this->input);
		
		return this->output;
	}

	double Backward(const VectorXd &delta,const int &delta_index)
	{
		const auto nw = 0.1;
		const auto nb = 0.1;
		
		auto diff = MathPlus::Differential(this->input, this->func);
		auto dot = delta.dot(this->weight);
		double ret_delta = dot*diff;

		VectorXd dw = nw*delta*this->output;
		double db = nb*delta(delta_index);
		
		this->weight -= dw;
		this->bias -= db;


		return ret_delta;
	}
};