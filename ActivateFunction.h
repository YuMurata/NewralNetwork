#pragma once

#include<Eigen/Core>

struct ActivateFunction
{
	virtual double Func(const Eigen::VectorXd &vec,const Eigen::VectorXd::Index &index)const = 0;
	virtual double Diff(const double &x)const = 0;
	virtual Eigen::VectorXd Diff(const Eigen::VectorXd &vec, const Eigen::VectorXd::Index &index)const
	{
		Eigen::VectorXd ret = Eigen::VectorXd::Zero(vec.size());
		ret(index) = this->Diff(vec(index));

		return ret;
	}
};

struct Identify :public ActivateFunction
{
	double Func(const Eigen::VectorXd &vec, const Eigen::VectorXd::Index &index)const override
	{
		auto ret = vec(index);
		return ret;
	}

	double Diff(const double &x)const override
	{
		auto ret = 1;
		return ret;
	}
};

struct ReLu :public ActivateFunction
{
	double Func(const Eigen::VectorXd &vec, const Eigen::VectorXd::Index &index)const override
	{
		auto ret = std::max(0., vec(index));;
		return ret;
	}

	double Diff(const double &x)const override
	{
		auto ret = x<0?0:1;
		return ret;
	}
};

struct Sigmoid :public ActivateFunction
{
	double Func(const Eigen::VectorXd &vec, const Eigen::VectorXd::Index &index)const override
	{
		auto ret = 1. / (1 + exp(-vec(index)));
		return ret;
	}

	double Diff(const double &x)const override
	{
		auto ret = x*(1.-x);
		return ret;
	}
};

struct Tanh :public ActivateFunction
{
	double Func(const Eigen::VectorXd &vec, const Eigen::VectorXd::Index &index)const override
	{
auto ret = tanh(vec(index));
	return ret;
	}

	double Diff(const double &x)const override
	{
		auto ret = 1.-sqrt(x);
		return ret;
	}
};

struct Softmax : public ActivateFunction 
{
	double Func(const Eigen::VectorXd &vec,const Eigen::VectorXd::Index &index) const override 
	{
		double alpha = vec.maxCoeff();
		auto numer = std::exp(vec[index] - alpha);
		auto denom=(vec.array() - alpha).array().exp().sum();
		auto ret = numer / denom;

		return ret;
	}

	double Diff(const double &x) const override
	{
		auto ret= x * (1.0 - x);
		return ret;
	}

	virtual Eigen::VectorXd Diff(const Eigen::VectorXd &vec,const Eigen::VectorXd::Index &index) const override
	{
		auto vec_size = vec.size();
		
		Eigen::VectorXd ret=Eigen::VectorXd::Zero(vec_size);
		
		for (size_t i = 0; i < vec_size; ++i)
		{
			ret(i) = (i == index) ? this->Diff(vec(index)) : -vec(i) * vec(index);
		}
		return ret;
	}
};
