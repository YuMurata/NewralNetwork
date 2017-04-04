#include"LayerImpl.h"
#include"ActivateFunction.h"

using namespace Eigen;
using namespace std;

Layer::Impl::Impl(const int &input_num, const int &output_num,unique_ptr<ActivateFunction> &func)
	:input_num(input_num), output_num(output_num), func(move(func))
{
	this->weight = MatrixXd::Random(input_num, output_num);
	this->bias = VectorXd::Random(output_num);

	this->input.resize(input_num);
	
	this->conversion.resize(output_num);
	this->output.resize(output_num);
}

Layer::Impl::Impl(Impl &impl)
{
	this->bias = impl.bias;
	this->conversion = impl.conversion;
	this->drop_mask = impl.drop_mask;
	
	this->func = move(impl.func);
	
	this->input = impl.input;
	this->input_num = impl.input_num;
	this->output = impl.output;
	this->output_num = impl.output_num;
	this->weight = impl.weight;
}