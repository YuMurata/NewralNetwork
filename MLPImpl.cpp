#include"MLPImpl.h"
#include"ActivateFunction.h"
#include"LossFunction.h"

using namespace std;

MLP::Impl::Impl(Params &params)
{
	srand(time_t(NULL));

	auto net_size = size(params.first);
	this->network.reserve(net_size);

	auto &layer_info = params.first;
	auto &loss = params.second;

	for (int i = 1; i < net_size; ++i)
	{
		auto input_num = layer_info[i-1].first;
		auto output_num = layer_info[i].first;
		auto func=move(layer_info[i-1].second);
		Layer layer(input_num, output_num, func);

		this->network.push_back(layer);
	}

	this->loss = move(loss);
}