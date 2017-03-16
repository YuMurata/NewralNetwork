#pragma once

#include"Newron.h"
#include"Randomer.h"

class Layer
{
public:
	vector<Newron> layer;
	int input_num;
	int output_num;

	VectorXd input;
	VectorXd conversion;
	VectorXd output;

	MatrixXd weight;
	VectorXd bias;
	
	Newron::ActFunc func;

	VectorXd drop_mask;

	Layer(const int &input_num, const int &output_num, const Newron::ActFunc &func)
		:input_num(input_num), output_num(output_num),func(func)
	{
		this->weight = MatrixXd::Random(input_num, output_num);
		this->bias = VectorXd::Random(output_num);

		this->input.resize(input_num);

		this->conversion.resize(output_num);
		this->output.resize(output_num);

		this->InitDrop();
	}

	//newronÇçlÇ¶ÇΩèÍçá
	/*Layer(const int &input_num, const int &output_num, const Newron::ActFunc &func)
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
	}*/

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
		this->input = input;
		for (int i = 0; i < this->output_num; ++i)
		{
			VectorXd col = this->weight.col(i);
			this->conversion(i) = col.dot(this->input)+this->bias(i);
			this->output(i) = this->func(this->conversion(i));
		}

		this->output.array() *= this->drop_mask.array();

		return this->output;
	}

	//newronÇçlÇ¶ÇΩèÍçá
	/*VectorXd Forward(const VectorXd &input)
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
	}*/

	VectorXd Backward(const VectorXd &deltas)
	{
		const double nw = 0.1;
		const double nb = 0.01;

		VectorXd mask_deltas = deltas.array()*this->drop_mask.array();


		MatrixXd dw = MatrixXd::Zero(this->input_num, this->output_num);
		for (int i = 0; i < this->output_num; ++i)
		{
			auto diff=MathPlus::Differential(this->conversion(i),this->func);
			auto delta = mask_deltas(i)*diff;
			dw.col(i) = nw*delta*this->input;
		}

		VectorXd db = nb*mask_deltas;

		this->weight -= dw;
		this->bias -= db;

		VectorXd ret = VectorXd::Zero(this->input_num);
		for (int i = 0; i < this->input_num; ++i)
		{
			VectorXd row = this->weight.row(i);
			auto dot = row.dot(mask_deltas);
			ret(i) = dot;
		}

		//return VectorXd::Zero(1);
		return ret;
	}

	//newron
	/*VectorXd Backward(const VectorXd &deltas)
	{
		VectorXd ret = VectorXd::Zero(this->input_num);
		VectorXd mask_delta=deltas.array()*this->drop_mask.array();
		
		for (int i = 0; i<this->output_num; ++i)
		{
			ret(i)= this->layer[i].Backward(mask_delta,i);
		}
		
		return ret;
	}*/
};
