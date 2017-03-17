#pragma once

#include"MLP.h"
#include"Layer.h"

struct MLP::Impl
{
	std::vector<Layer> network;
	
	Impl(Params &params);
};