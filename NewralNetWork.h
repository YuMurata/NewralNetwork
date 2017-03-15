#pragma once

#include"Layer.h"

//Data = pair<VectorXd, VectorXd>::first = input, second = teach
class NewralNetWork
{
public:
	using OutFunc = function<VectorXd(const VectorXd &)>;
	using LayerParam = pair<int, Newron::ActFunc>;
	using LayerParams = vector<LayerParam>;
	using Params = pair<LayerParams,OutFunc>;
	
	using Data = pair<VectorXd, VectorXd>;
	using DataList = vector<Data>;

	vector<Layer> network;
	OutFunc out_func;

	NewralNetWork(const Params &params)
		:out_func(params.second)
	{
		srand(time(NULL));

		auto net_size = size(params.first);
		this->network.reserve(net_size);
		for (int i = 0; i<net_size - 1; ++i)
		{
			auto input_num = params.first[i].first;
			auto output_num = params.first[i + 1].first;
			auto func = params.first[i].second;
			Layer layer(input_num, output_num, func);

			this->network.push_back(layer);
		}
	}

	VectorXd Forward(const VectorXd &input)
	{
		VectorXd output;
		VectorXd new_input = input;

		for (auto &i : this->network)
		{
			output = i.Forward(new_input);
			new_input = output;
		}

		VectorXd ret = this->out_func(output);

		return ret;
	}

	VectorXd Backward(const VectorXd &deltas)
	{
		VectorXd old_deltas = deltas;
		VectorXd new_deltas;

		reverse(begin(this->network), end(this->network));

		for (auto &i : this->network)
		{
			new_deltas = i.Backward(old_deltas);
			old_deltas = new_deltas;
		}

		reverse(begin(this->network), end(this->network));

		return new_deltas;
	}

	void Learn(const DataList &data_list, const double &threshold=1e-3)
	{
	
		auto learn_num = size(data_list);
		for (int i=0;i<learn_num;++i)
		{
			cout << i << endl;

			VectorXd input = data_list[i].first;
			VectorXd t = data_list[i].second;

			auto drop_func = [](Layer &x)
			{
				x.MakeDrop();
			};
			for_each(begin(this->network), end(this->network) - 1, drop_func);

			VectorXd output=this->Forward(input);
			VectorXd deltas = output - t;
			double MSE = deltas.array().square().sum()*0.5;
			
			while (MSE > threshold)
			{
				this->Backward(deltas);
				output=this->Forward(input);
				deltas = output - t;
				MSE = deltas.array().square().sum()*0.5;
			}
		}

		for (auto &i : this->network)
		{
			i.InitDrop();
		}
	}
};