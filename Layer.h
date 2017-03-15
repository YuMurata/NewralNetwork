#pragma once

#include"Newron.h"
#include"Randomer.h"

class Layer
{
public:
	vector<Newron> layer;
	int input_num;
	int output_num;

	VectorXd drop_mask;

	Layer(const int &input_num, const int &output_num, const Newron::ActFunc &func)
		:input_num(input_num), output_num(output_num)
	{
		auto gen_func = [&input_num,&func]()
		{
			auto ret = Newron(input_num, func);
			return ret;
		};

		this->layer.reserve(output_num);
		generate_n(back_inserter(this->layer), output_num, gen_func);

		this->InitDrop();
	}

	void MakeDrop()
	{
		Lottery<int> lottery(0,this->output_num);

		this->InitDrop();

		for (int i = 0; i < this->output_num / 2; ++i)
		{
			assert(lottery.Cast());
			auto index = lottery.Get();
			this->drop_mask(index) = 0;
		}
	}

	void InitDrop()
	{
		this->drop_mask = VectorXd::Ones(this->output_num);
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
		ret.array()*= this->drop_mask.array();
		return ret;
	}

	VectorXd Backward(const VectorXd &deltas)
	{
		VectorXd ret = VectorXd::Zero(this->input_num);
		VectorXd mask_delta=deltas.array()*this->drop_mask.array();
		
		for (int i = 0; i<this->output_num; ++i)
		{
			ret += this->layer[i].Backward(mask_delta(i));
		}
		
		return ret;
	}
};
