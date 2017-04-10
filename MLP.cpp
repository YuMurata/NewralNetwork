#include"MLPImpl.h"
#include"ActivateFunction.h"
#include"LossFunction.h"

#include<StringPlus.h>

using namespace Eigen;
using namespace std;

MLP::MLP(Params &params)
	:pimpl(new Impl(params)) {}

MLP::MLP(const string &file_name)
{
	this->LoadFile(file_name);
}

MLP::~MLP() = default;

MLP::Output MLP::Forward(const VectorXd &input)
{
	VectorXd output;
	VectorXd new_input = input;

	for (auto &i : this->pimpl->network)
	{
		output = i.Forward(new_input);
		new_input = output;
	}

	auto output_num = output.size();
	VectorXd ret =VectorXd::Zero(output_num);

	for (int i = 0; i < output_num; ++i)
	{
		ret(i) = this->pimpl->out->Func(output, i);
	}

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

void MLP::Disp()const
{
	auto net_size = size(this->pimpl->network);
	
	for (int i = 0; i < net_size; ++i)
	{
		cout << i << ":" << endl;
		this->pimpl->network[i].Disp();
	}
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

		Output output = this->Forward(input);
		LossFunction::Deltas deltas = this->pimpl->loss->Diff(output, t);
		LossFunction::E E = this->pimpl->loss->Func(output, t);

		while (E > threshold)
		{
			this->Backward(deltas);
			output = this->Forward(input);
			deltas = this->pimpl->loss->Diff(output, t);
			E = this->pimpl->loss->Func(output, t);
		}
	}

	for (auto &i : this->pimpl->network)
	{
		i.InitDrop();
	}
}

bool MLP::LoadFile(const string &file_name)
{
	this->pimpl = make_unique<Impl>(Params());

	auto data_list = this->PreLoad(file_name);

	auto itr = begin(data_list[0]);

	auto network_size = stoi(*itr);
	this->pimpl->network.reserve(network_size);

	++itr;

	auto func = [](const string &str)
	{
		Layer ret(str);
		return ret;
	};
	transform(itr, end(data_list[0]), back_inserter(this->pimpl->network),func);
	
	return true;
}

bool MLP::WriteFile(const string &file_name)
{
	auto network_size = size(this->pimpl->network);
	auto extension = ".txt";
	auto sub_names = Split(file_name, '.');
	auto sub_name = sub_names.front();

	enum offset
	{
		NETWORK_SIZE, OFFSET_NUM
	};

	vector<string> data_list;
	data_list.reserve(network_size+OFFSET_NUM);

	data_list.push_back(to_string(network_size));

	for (int i = 0; i < network_size; ++i)
	{
		auto serial = to_string(i);
		auto sub_file_name = sub_name + serial + extension;
		data_list.push_back(sub_file_name);
		this->pimpl->network[i].WriteFile(sub_file_name);
	}

	auto ret = this->PreWrite(file_name, Filer::DataList{ data_list });

	return ret;
}
