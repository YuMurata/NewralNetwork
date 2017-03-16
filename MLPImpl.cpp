#include"MLPImpl.h"

using namespace std;

MLP::Impl::Impl(const Params &params)
	:out_func(params.second)
{
	srand(time_t(NULL));

	auto net_size = size(params.first);
	this->network.reserve(net_size);
	for (int i = 0; i < net_size - 1; ++i)
	{
		auto input_num = params.first[i].first;
		auto output_num = params.first[i + 1].first;
		auto func = params.first[i].second;
		Layer layer(input_num, output_num, func);

		this->network.push_back(layer);
	}
}