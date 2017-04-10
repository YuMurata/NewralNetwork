#pragma once

#include<Eigen/Core>

struct LossFunction
{
	using Output = Eigen::VectorXd;
	using Target = Eigen::VectorXd;

	using E = double;
	using Delta = double;
	using Deltas = Eigen::VectorXd;

	virtual E Func(const Output &output, const Target &target) = 0;

	virtual Deltas Diff(const Output &output, const Target &target)
	{
		Deltas ret = output - target;
		return ret;
	}

	virtual Delta Diff(const Output &output, const Target &target, const int &index)
	{
		Deltas diff = this->Diff(output, target);
		auto ret = diff(index);

		return ret;
	}
};

struct MSE :public LossFunction
{
	E Func(const Output &output,const Target &target)
	{
		auto ret = 0.5*(output - target).array().square().sum();
		return ret;
	}
};

struct Cross :public LossFunction
{
	E Func(const Output &output, const Target &target)
	{
		auto ret = -(target.array()*output.array().log10() + (1 - target.array())*(1 - output.array()).log10()).sum();
		return ret;
	}
};

struct SoftCross :public LossFunction
{
	E Func(const Output &output, const Target &target)
	{
		auto ret = -(target.array()*output.array().log10()).sum();
		return ret;
	}
};