#include<iostream>
#include "opencv2/core/core.hpp"
using namespace std;
int main()
{

	cout << "hello" << endl;
	cv::Mat H = cv::findHomography(points1, points2, CV_RANSAC, 5 );

}
