// NewralNetWork.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include"NewralNetWork.h"
#include<Randomer.h>
#include"ActiveFunction.h"
#include"OutputFunction.h"

int main()
{
	int learn_num = 100;

	NewralNetWork::DataList data_list(learn_num);
	Randomer obj;

	auto making = [&obj]()
	{
		NewralNetWork::Data data;
		double a = obj.rand<uniform_int_distribution<>>(0, 1);
		double b = obj.rand<uniform_int_distribution<>>(0, 1);

		auto c = a + obj.rand<uniform_real_distribution<>>(-0.5, 0.5);
		auto d = a + obj.rand<uniform_real_distribution<>>(-0.5, 0.5);

		data.first =VectorXd(2);
		data.first << c, d;
		data.second =VectorXd(2);
		data.second << a, b;
		return data;
	};

	generate(begin(data_list), end(data_list), making);

	NewralNetWork::LayerParams layer_params =
	{
		make_pair(2,ReLu),
		make_pair(3,ReLu),
		make_pair(4,ReLu),
		make_pair(2,ReLu),
	};
	Eigen::VectorXd x;
	NewralNetWork::Params params(make_pair(layer_params, Real));

	NewralNetWork network(params);

	network.Learn(data_list);

	cout << "your input:" << endl;
	double a, b;
	while (cin >> a >> b)
	{
		VectorXd data(2);
		data << a, b;
		cout << "output" << endl;
		cout<<network.Forward(data)<<endl;
		cout << "your input:" << endl;
	}

	_getch();
    return 0;
}

