#pragma once

#include"MLP.h"
#include"Layer.h"

struct LossFunction;
struct ActivateFunction;

struct MLP::Impl
{
	std::vector<Layer> network;

	std::unique_ptr<ActivateFunction> out;
	std::unique_ptr<LossFunction> loss;

	Impl(Params &params);
};