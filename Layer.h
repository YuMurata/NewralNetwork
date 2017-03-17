#pragma once
#include"ActivateFunction.h"
struct ActivateFunction;

class Layer
{
private:
	struct Impl;
	std::shared_ptr<Impl> pimpl;

public:
	Layer(const int &input_num, const int &output_num, std::unique_ptr<ActivateFunction> &func);
	
	~Layer();

	void MakeDrop();

	void InitDrop();

	Eigen::VectorXd Forward(const Eigen::VectorXd &input);

	Eigen::VectorXd Backward(const Eigen::VectorXd &deltas);

};
