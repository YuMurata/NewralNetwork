#pragma once

#include"Layer.h"

//Data = pair<VectorXd, VectorXd>::first = input, second = teach
class MLP
{
private:
	struct Impl;
	std::unique_ptr<Impl> pimpl;

public:
	using OutFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
	using LayerParam = std::pair<int, Layer::ActFunc>;
	using LayerParams = std::vector<LayerParam>;
	using Params = std::pair<LayerParams,OutFunc>;
	
	using Data = std::pair<Eigen::VectorXd, Eigen::VectorXd>;
	using DataList = std::vector<Data>;

	MLP(const Params &params);

	~MLP();

	Eigen::VectorXd Forward(const Eigen::VectorXd &input);

	Eigen::VectorXd Backward(const Eigen::VectorXd &deltas);

	void Learn(const DataList &data_list, const double &threshold = 1e-3);
};