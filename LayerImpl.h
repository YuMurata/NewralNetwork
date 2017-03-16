#pragma once

#include"Layer.h"

struct Layer::Impl
{
	int input_num;
	int output_num;

	Eigen::VectorXd input;
	Eigen::VectorXd conversion;
	Eigen::VectorXd output;

	Eigen::MatrixXd weight;
	Eigen::VectorXd bias;

	ActFunc func;

	Eigen::VectorXd drop_mask;

	Impl(const int &input_num, const int &output_num, const ActFunc &func);
};