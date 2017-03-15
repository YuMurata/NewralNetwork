#pragma once

VectorXd SoftMax(const VectorXd &x)
{
	auto x_size = x.size();
	VectorXd ret(x_size);

	auto sum = x.sum();

	for (int i = 0; i < x_size; ++i)
	{
		ret(i) = x(i) / sum;
	}

	return ret;
};

VectorXd OneHot(const VectorXd &x)
{
	VectorXd::Index index;
	x.maxCoeff(&index);

	VectorXd ret=VectorXd::Zero(x.size());
	ret(index) = 1;

	return ret;
};

VectorXd Real(const VectorXd &x)
{
	VectorXd ret = x;
	return ret;
};