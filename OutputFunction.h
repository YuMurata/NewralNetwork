#pragma once

Eigen::VectorXd SoftMax(const Eigen::VectorXd &x)
{
	auto x_size = x.size();
	Eigen::VectorXd ret(x_size);

	auto sum = x.sum();

	for (int i = 0; i < x_size; ++i)
	{
		ret(i) = x(i) / sum;
	}

	return ret;
};

Eigen::VectorXd OneHot(const Eigen::VectorXd &x)
{
	Eigen::VectorXd::Index index;
	x.maxCoeff(&index);

	Eigen::VectorXd ret=Eigen::VectorXd::Zero(x.size());
	ret(index) = 1;

	return ret;
};

Eigen::VectorXd Real(const Eigen::VectorXd &x)
{
	Eigen::VectorXd ret = x;
	return ret;
};