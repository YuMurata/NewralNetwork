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

	auto Real=[&obj]()
	{
		MLP::Data data;
		double a = obj.rand<uniform_real_distribution<>>(0, 10);
		auto c = a;//+ obj.rand<uniform_real_distribution<>>(-0.5, 0.5);

		data.first = Map<MLP::Input>(&c,1);
		data.second = Map<MLP::Target>(&a,1);

		return data;
	};

	auto IsOr = [](const VectorXd &x)
	{
		auto sum=x.sum();
		auto flag = sum >= 1;

		auto ret = flag ? 1 : 0;
		return ret;
	};

	auto Or = [&obj,&IsOr]()
	{
		MLP::Data data;
		double a = obj.rand<uniform_int_distribution<>>(0, 1);
		double b = obj.rand<uniform_int_distribution<>>(0, 1);

		data.first = VectorXd(2);
		data.first << a, b;
		data.second = VectorXd(1);
		data.second << IsOr(data.first);

		return data;
	};
	
	generate(begin(data_list), end(data_list), Real);

	MLP::Params params;
	
	auto &layer_info = params.first;
	auto &loss = params.second;

	auto input_num = 1;

	layer_info.push_back(make_pair(input_num, move(make_unique<ReLu>())));
	layer_info.push_back(make_pair(3, move(make_unique<ReLu>())));
	layer_info.push_back(make_pair(1, move(make_unique<ReLu>())));

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
		cin.clear();
		MLP::Input data=Map<MLP::Input>(input.data(),input_num);
		
		VectorXd output = network.Forward(data);
		cout << "output" << endl;
		cout<<output<<endl;

		cout << "t" << endl;
		cout << data << endl;

		cout << "your input:" << endl;
	}

	_getch();
    return 0;
}

