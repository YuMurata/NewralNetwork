#include"LayerImpl.h"

using namespace Eigen;
using namespace std;

Layer::Impl::Impl(const int &input_num, const int &output_num, const ActFunc &func)
	:input_num(input_num), output_num(output_num), func(func)
{
	this->weight = MatrixXd::Random(input_num, output_num);
	this->bias = VectorXd::Random(output_num);

	this->input.resize(input_num);

	this->conversion.resize(output_num);
	this->output.resize(output_num);
}