#pragma once

#include"Layer.h"

struct ActivateFunction;

//Data = pair<VectorXd, VectorXd>::first = input, second = teach
class MLP
{
private:
	struct Impl;
	std::unique_ptr<Impl> pimpl;

public:
	using Param = std::pair<const int, std::unique_ptr<ActivateFunction>>;
	using Params = std::vector<Param>;
	
	using Data = std::pair<Eigen::VectorXd, Eigen::VectorXd>;
	using DataList = std::vector<Data>;

	MLP(Params &params);

	~MLP();

	Eigen::VectorXd Forward(const Eigen::VectorXd &input);

	Eigen::VectorXd Backward(const Eigen::VectorXd &deltas);

	void Learn(const DataList &data_list, const double &threshold = 1e-3);
};