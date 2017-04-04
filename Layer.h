#pragma once

#include<Eigen/Core>
#include<Filer.h>
#include<memory>

struct ActivateFunction;

class Layer:public Filer
{
private:
	struct Impl;
	std::unique_ptr<Impl> pimpl;

public:
	Layer(const int &input_num, const int &output_num, std::unique_ptr<ActivateFunction> &func);
	
	Layer(const std::string &file_name);

	Layer(const Layer &layer);

	~Layer();

	Layer& operator=(const Layer &layer);
	
	void MakeDrop();

	void InitDrop();

	Eigen::VectorXd Forward(const Eigen::VectorXd &input);

	Eigen::VectorXd Backward(const Eigen::VectorXd &deltas);

	void Disp()const;

#ifdef UNICODE
	bool LoadFile(const std::string &file_name)override;

	bool WriteFile(const std::string &file_name)override;
#else
	bool LoadFile(const std::wstring &file_name)override
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

	bool WriteFile(const std::wstring &file_name)override
	{
		vector<vector<wstring>> data_list;
		this->write(this->q_table, &data_list);

		this->PreWrite(file_name, data_list);

		return true;
	}
#endif
};
