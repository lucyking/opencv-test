#include <iostream>
#include <cv.h>
#include <highgui.h>

using namespace cv;

int main(int argc, char* argv[])
{

	Mat im = imread("src.bmp");				// 入力画像の取得
	Mat tmp = imread("target-raw.bmp");				// テンプレート画像の取得
	//Mat tmp = imread("target.jpg");				// テンプレート画像の取得
	Mat match;
	if (!im.data || !tmp.data) {
		printf("FILED!!\n");
		return -1;		// エラー処理
	}
		
	// テンプレートマッチングで探索
	matchTemplate(im, tmp, match, CV_TM_CCOEFF_NORMED);

	Point pt;
	// 最大のスコア(テンプレート画像と最も類似する)領域の座標を取得
	minMaxLoc(match, 0, 0, 0, &pt);
	// 最もマッチした領域を赤い矩形で囲む
	rectangle(im, pt, Point(pt.x + tmp.cols, pt.y + tmp.rows), Scalar(0, 0, 255), 2, 8, 0);
	imshow("Show image", im);					// 画像の表示
	waitKey(0);									// キー入力待機
	return 0;
}

