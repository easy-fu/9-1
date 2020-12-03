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
		cout << "��ͼ������·" << endl;
		destroyAllWindows();
		return -1;
	}

	int cell_size = 16;                  //16*16��cell
	int angle_num = 8;                   //�Ƕ�����Ϊ8

	//ͼ��ָ�Ϊy_cellnum�У�x_cellnum��
	int x_num = refImg.cols / cell_size;
	int y_num = refImg.rows / cell_size;
	int bins = x_num * y_num * angle_num;//���鳤��

	//���ٶ�̬����
	float* ref_hog = new float[bins];
	memset(ref_hog, 0, sizeof(float) * bins);
	float* img1_hog = new float[bins];
	memset(img1_hog, 0, sizeof(float) * bins);
	float* img2_hog = new float[bins];
	memset(img2_hog, 0, sizeof(float) * bins);

	//��������ͼ����ݶ�ֱ��ͼ
	hog_hisgram(refImg, ref_hog, cell_size, angle_num);
	hog_hisgram(Img1, img1_hog, cell_size, angle_num);
	hog_hisgram(Img2, img2_hog, cell_size, angle_num);

	//����ͼ������ƶ�
	float smlrt1, smlrt2;
	smlrt1 = Similarity(ref_hog, img1_hog, bins);
	smlrt2 = Similarity(ref_hog, img2_hog, bins);

	delete[] ref_hog;
	delete[] img1_hog;
	delete[] img2_hog;

	//�Ƚ����ƶ�
	if (smlrt1 > smlrt2)
		cout << "Img1��ο�ͼ����ӽ�" << endl;
	else
		cout << "Img2��ο�ͼ����ӽ�" << endl;
	waitKey(0);
}

void hog_hisgram(InputArray src, float* histogram, int cellsize, int anglenum)
{
	Mat gray, grd_x, grd_y;                           //�Ҷȣ�x�����y������ݶ�
	//���������ݶȵķ�ֵ�ͷ���
	cvtColor(src, gray, COLOR_BGR2GRAY);
	Mat angle, mag;                                   //�ݶȷ����ݶȷ�ֵ
	Sobel(gray, grd_x, CV_32F, 1, 0, 3);
	Sobel(gray, grd_y, CV_32F, 0, 1, 3);
	cartToPolar(grd_x, grd_y, mag, angle, true);

	//����cell�ĸ���
	//ͼ��ָ�Ϊy_cellnum�У�x_cellnum��
	int x_cellnum, y_cellnum;
	x_cellnum = gray.cols / cellsize;
	y_cellnum = gray.rows / cellsize;


	int angle_area = 360 / anglenum;                  //ÿ�����������������ĽǶ���
	//��ѭ��������cell
	for (int i = 0; i < y_cellnum; i++)
	{
		for (int j = 0; j < x_cellnum; j++)
		{
			//�������Ȥ����roi,ȡ��ÿ��cell
			Rect roi;
			roi.width = cellsize;
			roi.height = cellsize;
			roi.x = j * cellsize;
			roi.y = i * cellsize;

			Mat RoiAngle, RoiMag;
			RoiAngle = angle(roi);                    //ÿ��cell�е��ݶȷ���
			RoiMag = mag(roi);                        //ÿ��cell�е��ݶȷ�ֵ

			//����RoiAngel��RoiMat
			int head = (i * x_cellnum + j) * anglenum;//cell�ݶ�ֱ��ͼ�ĵ�һ��Ԫ������ֱ��ͼ�е�λ��
			for (int m = 0; m < cellsize; m++)
			{
				for (int n = 0; n < cellsize; n++)
				{
					int idx = ((int)RoiAngle.at<float>(m, n)) / angle_area;//���ݶ���������������
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
	return 1 / (1 + distance);//�������ƶ�
}