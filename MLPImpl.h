#pragma once

#include"MLP.h"
#include"Layer.h"

struct MLP::Impl
{
	std::vector<Layer> network;
	OutFunc out_func;

	Impl(const Params &params);
};