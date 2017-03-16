#include"MLPImpl.h"

using namespace Eigen;
using namespace std;

MLP::MLP(const Params &params)
	:pimpl(new Impl(params)) {}

MLP::~MLP() = default;

VectorXd MLP::Forward(const VectorXd &input)
{
	VectorXd output;
	VectorXd new_input = input;

	for (auto &i : this->pimpl->network)
	{
		output = i.Forward(new_input);
		new_input = output;
	}

	VectorXd ret = this->pimpl->out_func(output);

	return ret;
}

VectorXd MLP::Backward(const VectorXd &deltas)
{
	VectorXd old_deltas = deltas;
	VectorXd new_deltas;

	reverse(begin(this->pimpl->network), end(this->pimpl->network));

	for (auto &i : this->pimpl->network)
	{
		new_deltas = i.Backward(old_deltas);
		old_deltas = new_deltas;
	}

	reverse(begin(this->pimpl->network), end(this->pimpl->network));

	return new_deltas;
}

void MLP::Learn(const DataList &data_list, const double &threshold)
{

	auto learn_num = size(data_list);
	for (int i = 0; i<learn_num; ++i)
	{
		cout << i << endl;

		VectorXd input = data_list[i].first;
		VectorXd t = data_list[i].second;

		auto drop_func = [](Layer &x)
		{
			x.MakeDrop();
		};
		for_each(begin(this->pimpl->network), end(this->pimpl->network) - 1, drop_func);

		VectorXd output = this->Forward(input);
		VectorXd deltas = output - t;
		double MSE = deltas.array().square().sum()*0.5;

		while (MSE > threshold)
		{
			this->Backward(deltas);
			output = this->Forward(input);
			deltas = output - t;
			MSE = deltas.array().square().sum()*0.5;
		}
	}

	for (auto &i : this->pimpl->network)
	{
		i.InitDrop();
	}
}