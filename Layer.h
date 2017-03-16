#pragma once

class Layer
{
private:
	struct Impl;
	std::unique_ptr<Impl> pimpl;

public:

	using ActFunc = std::function<double(const double &)>;

	Layer(const int &input_num, const int &output_num, const ActFunc &func);
	
	Layer(const Layer &obj);

	~Layer();

	void MakeDrop();

	void InitDrop();

	Eigen::VectorXd Forward(const Eigen::VectorXd &input);

	Eigen::VectorXd Backward(const Eigen::VectorXd &deltas);

	Layer& operator=(const Layer &obj);
};
