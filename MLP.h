#pragma once

#include"Layer.h"
#include<Filer.h>
#include<string>

struct ActivateFunction;
struct LossFunction;
//Data = pair<VectorXd, VectorXd>::first = input, second = teach
class MLP:public Filer
{
private:
	struct Impl;
	std::unique_ptr<Impl> pimpl;

public:
	using Param = std::pair<const int, std::unique_ptr<ActivateFunction>>;
	using Params = std::pair<std::vector<Param>,std::unique_ptr<LossFunction>>;
	
	using Input = Eigen::VectorXd;
	using Output = Eigen::VectorXd;
	using Target = Eigen::VectorXd;

	using Data = std::pair<Input, Target>;
	using DataList = std::vector<Data>;

	MLP(Params &params);

	MLP(const std::string &file_name);

	~MLP();

	Output Forward(const Input &input);

	Eigen::VectorXd Backward(const Eigen::VectorXd &deltas);

	void Learn(const DataList &data_list, const double &threshold = 1e-3);

	void Disp()const;

#ifdef UNICODE
	bool LoadFile(const std::string &file_name)override;

	bool WriteFile(const std::string &file_name)override;
#else
	bool LoadFile(const wstring &file_name)override
	{
		auto data_list = this->PreLoad(file_name);

		auto Qs = this->load(data_list);

		for (auto &i : Qs)
		{
			//			this->q_table.emplace_hint(begin(this->q_table), i.first, i.second);
			this->q_table[i.first] = i.second;
		}

		return !data_list.empty();
	}

	bool WriteFile(const wstring &file_name)override
	{
		vector<vector<wstring>> data_list;
		this->write(this->q_table, &data_list);

		this->PreWrite(file_name, data_list);

		return true;
	}
#endif
};