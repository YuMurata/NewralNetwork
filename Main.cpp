// NewralNetWork.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include"MLP.h"
#include<Randomer.h>
#include"ActivateFunction.h"
#include"LossFunction.h"
#include"OutputFunction.h"
#include<Builder.h>

using namespace std;
using namespace Eigen;

int main()
{
	int learn_num = 100;

	MLP::DataList data_list(learn_num);
	Randomer obj;

	struct Generator:public Randomer
	{
		using Data = vector<double>;

		virtual int InputNum()=0;

		virtual int OutputNum() = 0;

		virtual Data Input() = 0;

		virtual Data Target(const Data &input) = 0;

		auto Gen()
		{
			auto input_num = this->InputNum();
			auto output_num = this->OutputNum();

			MLP::Data data;
			auto input = this->Input();
			auto target = this->Target(input);

			data.first = Map<MLP::Input>(input.data(), input_num);
			data.second = Map<MLP::Target>(target.data(), output_num);

			return data;
		}

		virtual MLP::Target Ans(const MLP::Input &input) = 0;
	};

	struct Sin:public Generator
	{
		int InputNum()override
		{
			auto ret = 1;
			return ret;
		}

		int OutputNum()override
		{
			auto ret = 1;
			return ret;
		}

		Data Input()override
		{
			auto input = this->rand<uniform_real_distribution<>>(0, 2 * 3.14);
			Data ret{ input };
			return ret;
		}

		Data Target(const Data &input)override
		{
			Data ret{ sin(input.front()) };
			return ret;
		}

		MLP::Target Ans(const MLP::Input &input)override
		{
			auto ret = input.array().sin();
			return ret;
		}
	};

	struct Real :public Generator
	{
		int InputNum()override
		{
			auto ret = 1;
			return ret;
		}

		int OutputNum()override
		{
			auto ret = 1;
			return ret;
		}

		Data Input()override
		{
			auto input = this->rand<uniform_real_distribution<>>(0, 10);
			Data ret{ input };
			return ret;
		}

		Data Target(const Data &input)override
		{
			Data ret{ input };
			return ret;
		}

		MLP::Target Ans(const MLP::Input &input)override
		{
			auto ret = input.array();
			return ret;
		}
	};

	struct Or :public Generator
	{
		int InputNum()override
		{
			auto ret = 2;
			return ret;
		}

		int OutputNum()override
		{
			auto ret = 1;
			return ret;
		}

		Data Input()override
		{
			auto input1 = 1.*this->rand<uniform_int_distribution<>>(0, 1);
			auto input2 = 1.*this->rand<uniform_int_distribution<>>(0, 1);
			Data ret{ input1,input2 };
			return ret;
		}

		Data Target(const Data &input)override
		{
			auto target = input[0] || input[1] ? 1. : 0.;
			Data ret{ target };
			return ret;
		}

		MLP::Target Ans(const MLP::Input &input)override
		{
			auto sum = input.sum();
			auto flag = sum >= 1;

			auto ans = flag ? 1 : 0;
			MLP::Target ret(1);
			ret << ans;
			return ret;
		}
	};
	
	unique_ptr<Generator> gen = make_unique<Sin>();
	auto func = [&gen]()
	{
		auto ret = gen->Gen();
		return ret;
	};

	generate(begin(data_list), end(data_list), func);

	MLP::Params params;
	
	auto &layer_info = params.first;
	auto &loss = params.second;

	auto input_num = gen->InputNum();
	auto output_num = gen->OutputNum();

	layer_info.push_back(make_pair(input_num, make_unique<ReLu>()));
	layer_info.push_back(make_pair(input_num, make_unique<ReLu>()));
	layer_info.push_back(make_pair(input_num, make_unique<Identify>()));
	layer_info.push_back(make_pair(output_num, make_unique<Identify>()));
	
	
	loss = make_unique<MSE>();

	MLP network(params);

	network.Learn(data_list);

	cout << "your input:" << endl;
	while (1)
	{
		vector<double> input;
		for (int i = 0; i < input_num; ++i)
		{
			double dum;
			cin >> dum;
			input.push_back(dum);
		}
	
		MLP::Input data=Map<MLP::Input>(input.data(),input_num);
		
		VectorXd output = network.Forward(data);
		cout << "output" << endl;
		cout<<output<<endl;

		cout << "t" << endl;
		cout << gen->Ans(data) << endl;

		cout << "your input:" << endl;
	}

	_getch();
    return 0;
}

