#include"MLPImpl.h"
#include"ActivateFunction.h"
using namespace std;

MLP::Impl::Impl(Params &params)
{
	srand(time_t(NULL));

	auto net_size = size(params);
	this->network.reserve(net_size);
	for (int i = 0; i < net_size - 1; ++i)
	{
		auto input_num = params[i].first;
		auto output_num = params[i + 1].first;
		auto func=move(params[i].second);
		Layer layer(input_num, output_num, func);

		this->network.push_back(layer);
	}
}