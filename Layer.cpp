#include"LayerImpl.h"
#include"ActivateFunction.h"

#include<Randomer.h>

using namespace Eigen;
using namespace std;

Layer::Layer(const int &input_num, const int &output_num,unique_ptr<ActivateFunction> &func)
	:pimpl(new Impl(input_num, output_num, func)) 
{
	this->InitDrop();
}

Layer::Layer(const string &file_name)
{
	this->LoadFile(file_name);
}

Layer::Layer(const Layer &layer)
{
	auto input = *layer.pimpl;
	this->pimpl=make_unique<Impl>(input);
}

//Layer::Layer(Layer &&layer)
//{
//	this->pimpl = move(layer.pimpl);
//}

Layer::~Layer() = default;

Layer& Layer::operator=(const Layer &layer)
{
	this->pimpl = make_unique<Impl>(*layer.pimpl);
	return *this;
}

void Layer::MakeDrop()
{
	Lottery<int> lottery(0, this->pimpl->output_num);

	this->InitDrop();

	for (int i = 0; i < this->pimpl->output_num / 2; ++i)
	{
		assert(lottery.Cast());
		auto index = lottery.Get();
		this->pimpl->drop_mask(index) = 0;
	}
}

void Layer::InitDrop()
{
	this->pimpl->drop_mask = VectorXd::Ones(this->pimpl->output_num);
}

VectorXd Layer::Forward(const VectorXd &input)
{
	this->pimpl->input = input;
	for (int i = 0; i < this->pimpl->output_num; ++i)
	{
		VectorXd col = this->pimpl->weight.col(i);
		this->pimpl->conversion(i) = col.dot(this->pimpl->input) + this->pimpl->bias(i);
		this->pimpl->output(i) = this->pimpl->func->Func(this->pimpl->conversion,i);
	}

	this->pimpl->output.array() *= this->pimpl->drop_mask.array();

	return this->pimpl->output;
}

VectorXd Layer::Backward(const VectorXd &deltas)
{
	const double nw = 0.1;
	const double nb = 0.01;

	VectorXd mask_deltas = deltas.array()*this->pimpl->drop_mask.array();


	MatrixXd dw = MatrixXd::Zero(this->pimpl->input_num, this->pimpl->output_num);
	for (int i = 0; i < this->pimpl->output_num; ++i)
	{
		auto diff = this->pimpl->func->Diff(this->pimpl->conversion(i));
		auto delta = mask_deltas(i)*diff;
		dw.col(i) = nw*delta*this->pimpl->input;
	}

	VectorXd db = nb*mask_deltas;

	this->pimpl->weight -= dw;
	this->pimpl->bias -= db;

	VectorXd ret = VectorXd::Zero(this->pimpl->input_num);
	for (int i = 0; i < this->pimpl->input_num; ++i)
	{
		VectorXd row = this->pimpl->weight.row(i);
		auto dot = row.dot(mask_deltas);
		ret(i) = dot;
	}

	return ret;
}

void Layer::Disp()const
{
	cout << "weight:" << endl;
	cout << this->pimpl->weight<<endl;

	cout << "bias:" << endl;
	cout << this->pimpl->bias << endl;
}