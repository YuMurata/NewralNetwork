#include"LayerImpl.h"

#include"ActivateFunction.h"

#include<StringPlus.h>
#include<sstream>
#include<typeinfo>
#include<Builder.h>

using namespace Eigen;
using namespace std;

bool Layer::LoadFile(const string &file_name)
{
	this->pimpl = make_unique<Impl>(0, 0, unique_ptr<ActivateFunction>());

	auto data_list = this->PreLoad(file_name);

	auto itr = begin(data_list);

	auto row = stoi((*itr)[0]);
	auto col = stoi((*itr)[1]);

	this->pimpl->input_num = row;
	this->pimpl->output_num = col;

	++itr;
	
	this->pimpl->weight = MatrixXd::Zero(row, col);

	auto Trans = [](const int &size, const vector<vector<string>>::iterator &itr)->VectorXd
	{
		auto func = [](const string &input)
		{
			auto ret = stod(input);
			return ret;
		};

		vector<double> temp;
		temp.reserve(size);

		transform(begin(*itr), end(*itr), back_inserter(temp), func);

		VectorXd ret = Map<VectorXd>(temp.data(), size);

		return ret;
	};

	for (int i = 0; i < row; ++i, ++itr)
	{
		auto vec = Trans(col, itr);
		this->pimpl->weight.row(i) = vec;
	}

	auto bias_num = stoi((*itr)[0]);
	++itr;
	auto bias = Trans(bias_num, itr);
	this->pimpl->bias = bias;

	++itr;

	Builder<ActivateFunction> builder;
	builder.Register<Identify>(typeid(Identify).name());
	builder.Register<ReLu>(typeid(ReLu).name());
	builder.Register<Sigmoid>(typeid(Sigmoid).name());
	builder.Register<Tanh>(typeid(Tanh).name());
	builder.Register<Softmax>(typeid(Softmax).name());

	auto func_name = (*itr)[0];

	this->pimpl->func = builder.Create(func_name);

	return true;
}

bool Layer::WriteFile(const std::string &file_name)
{
	DataList data_list;

	auto Reserve = [&]()
	{
		enum offset
		{
			ROW_NUM, BIAS_NUM, BIAS, OFFSET_NUM,
		};

		auto size = this->pimpl->weight.rows() + OFFSET_NUM;
		data_list.reserve(size);
	};

	auto WriteWeight = [&]()
	{
		auto row = to_string(this->pimpl->weight.rows());
		auto col = to_string(this->pimpl->weight.cols());

		data_list.push_back(vector<string>{row, col});

		stringstream weight;
		weight << this->pimpl->weight;

		string line;
		while (getline(weight, line))
		{
			auto input = Split(line, ' ');
			data_list.push_back(input);
		}
	};

	auto WriteBias = [&]()
	{
		auto bias_size = to_string(this->pimpl->bias.rows());
		data_list.push_back(vector<string>{bias_size});

		stringstream bias;
		bias << this->pimpl->bias.transpose();

		string line;
		while (getline(bias, line))
		{
			auto input = Split(line, ' ');
			data_list.push_back(input);
		}
	};

	auto WriteFunc = [&]()
	{
		auto func_name = typeid(*this->pimpl->func).name();
		data_list.push_back(vector<string>{func_name});
	};

	Reserve();
	WriteWeight();
	WriteBias();
	WriteFunc();

	auto ret = this->PreWrite(file_name, data_list);

	return ret;
}