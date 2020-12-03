#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;


void hog_hisgram(InputArray src, float* histogram, int cellsize, int anglenum);

float Similarity(float* hist1, float* hist2, int length);

int main()
{
	Mat refImg = imread("C:/Users/DELL/Desktop/6.jpg");
	Mat Img1 = imread("C:/Users/DELL/Desktop/7.jpg");
	Mat Img2 = imread("C:/Users/DELL/Desktop/8.jpg");

	if (refImg.empty() || Img1.empty() || Img2.empty())
	{
		cout << "打开图像发生错路" << endl;
		destroyAllWindows();
		return -1;
	}

	int cell_size = 16;                  //16*16的cell
	int angle_num = 8;                   //角度量化为8

	//图像分割为y_cellnum行，x_cellnum列
	int x_num = refImg.cols / cell_size;
	int y_num = refImg.rows / cell_size;
	int bins = x_num * y_num * angle_num;//数组长度

	//开辟动态数组
	float* ref_hog = new float[bins];
	memset(ref_hog, 0, sizeof(float) * bins);
	float* img1_hog = new float[bins];
	memset(img1_hog, 0, sizeof(float) * bins);
	float* img2_hog = new float[bins];
	memset(img2_hog, 0, sizeof(float) * bins);

	//计算三幅图像的梯度直方图
	hog_hisgram(refImg, ref_hog, cell_size, angle_num);
	hog_hisgram(Img1, img1_hog, cell_size, angle_num);
	hog_hisgram(Img2, img2_hog, cell_size, angle_num);

	//计算图像的相似度
	float smlrt1, smlrt2;
	smlrt1 = Similarity(ref_hog, img1_hog, bins);
	smlrt2 = Similarity(ref_hog, img2_hog, bins);

	delete[] ref_hog;
	delete[] img1_hog;
	delete[] img2_hog;

	//比较相似度
	if (smlrt1 > smlrt2)
		cout << "Img1与参考图像最接近" << endl;
	else
		cout << "Img2与参考图像最接近" << endl;
	waitKey(0);
}

void hog_hisgram(InputArray src, float* histogram, int cellsize, int anglenum)
{
	Mat gray, grd_x, grd_y;                           //灰度，x方向和y方向的梯度
	//计算像素梯度的幅值和方向
	cvtColor(src, gray, COLOR_BGR2GRAY);
	Mat angle, mag;                                   //梯度方向，梯度幅值
	Sobel(gray, grd_x, CV_32F, 1, 0, 3);
	Sobel(gray, grd_y, CV_32F, 0, 1, 3);
	cartToPolar(grd_x, grd_y, mag, angle, true);

	//计算cell的个数
	//图像分割为y_cellnum行，x_cellnum列
	int x_cellnum, y_cellnum;
	x_cellnum = gray.cols / cellsize;
	y_cellnum = gray.rows / cellsize;


	int angle_area = 360 / anglenum;                  //每个量化级数所包含的角度数
	//外循环，遍历cell
	for (int i = 0; i < y_cellnum; i++)
	{
		for (int j = 0; j < x_cellnum; j++)
		{
			//定义感兴趣区域roi,取出每个cell
			Rect roi;
			roi.width = cellsize;
			roi.height = cellsize;
			roi.x = j * cellsize;
			roi.y = i * cellsize;

			Mat RoiAngle, RoiMag;
			RoiAngle = angle(roi);                    //每个cell中的梯度方向
			RoiMag = mag(roi);                        //每个cell中的梯度幅值

			//遍历RoiAngel和RoiMat
			int head = (i * x_cellnum + j) * anglenum;//cell梯度直方图的第一个元素在总直方图中的位置
			for (int m = 0; m < cellsize; m++)
			{
				for (int n = 0; n < cellsize; n++)
				{
					int idx = ((int)RoiAngle.at<float>(m, n)) / angle_area;//该梯度所处的量化级数
					histogram[head + idx] += RoiMag.at<float>(m, n);
				}
			}
		}
	}

}

float Similarity(float* hist1, float* hist2, int length)
{
	float sum = 0;
	float distance;
	for (int i = 0; i < length; i++)
	{
		sum += pow(hist1[i] - hist2[i], 2);
	}
	distance = sqrt(sum);
	return 1 / (1 + distance);//返回相似度
}