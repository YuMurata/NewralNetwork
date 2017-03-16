#pragma once


double Identify(const double &x)
{
	auto ret = x;
	return ret;
}

double ReLu(const double &x)
{
	auto ret = std::max(0., x);
	return ret;
}

double Sigmoid(const double &x)
{
	auto ret = 1. / (1 + exp(-x));
	return ret;
}

double Tanh(const double &x)
{
	auto ret = tanh(x);
	return ret;
}
