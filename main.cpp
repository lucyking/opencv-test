void matchTemplate( const Mat& _img, const Mat& _templ, Mat& result, int method )
{
    CV_Assert( CV_TM_SQDIFF <= method && method <= CV_TM_CCOEFF_NORMED );
    //numType用来表示模板匹配的方式，0表示相关匹配法，1表示相关系数匹配法，2表示平方差匹配法
    //isNormed表示是否进行归一化处理，true表示进行归一化，false表示不进行归一化处理
    int numType = method == CV_TM_CCORR || method == CV_TM_CCORR_NORMED ? 0 :
                  method == CV_TM_CCOEFF || method == CV_TM_CCOEFF_NORMED ? 1 : 2;
    bool isNormed = method == CV_TM_CCORR_NORMED ||
                    method == CV_TM_SQDIFF_NORMED ||
                    method == CV_TM_CCOEFF_NORMED;
    //判断两幅图像的大小关系，如果输入的原始图像比匹配图像要小，则将原始图像作为模板，原来的模板图像作为搜索图
    Mat img = _img, templ = _templ;
    if( img.rows < templ.rows || img.cols < templ.cols )
        std::swap(img, templ);
    
    CV_Assert( (img.depth() == CV_8U || img.depth() == CV_32F) &&
               img.type() == templ.type() );

   //crossCorr函数是将输入图像做了一次DFT变换（离散傅里叶变换），将空间域的图像转换到频率域中来进行处理，并将处理的结果存放在result中
    int cn = img.channels();
    crossCorr( img, templ, result,
               Size(img.cols - templ.cols + 1, img.rows - templ.rows + 1),
               CV_32F, Point(0,0), 0, 0);

    //如果是相关匹配方法，此处已经计算完毕，返回
    if( method == CV_TM_CCORR )
        return;

    //将模板看作单位1，计算每一个像元所占的百分比（也可以理解为整个模板面积为1，计算每个像元的面积）
    double invArea = 1./((double)templ.rows * templ.cols);

    Mat sum, sqsum;
    Scalar templMean, templSdv;
    double *q0 = 0, *q1 = 0, *q2 = 0, *q3 = 0;
    double templNorm = 0, templSum2 = 0;

    //相关系数匹配算法
    if( method == CV_TM_CCOEFF )
    {
        integral(img, sum, CV_64F);//对原始图像进行求和
        templMean = mean(templ);//计算模板图像的均值向量
    }
    else//其他匹配算法
    {
        integral(img, sum, sqsum, CV_64F);//计算原始图像的和以及平方和
        meanStdDev( templ, templMean, templSdv );//计算模板图像的均值向量和方差向量

        templNorm = CV_SQR(templSdv[0]) + CV_SQR(templSdv[1]) +
                    CV_SQR(templSdv[2]) + CV_SQR(templSdv[3]);//计算所有通道的方差和

        if( templNorm < DBL_EPSILON && method == CV_TM_CCOEFF_NORMED )
        {//如果所有通道的方差的和等于0，并且使用的方法是归一化相关系数匹配方法，则返回
            result = Scalar::all(1);
            return;
        }
        
        templSum2 = templNorm +
                     CV_SQR(templMean[0]) + CV_SQR(templMean[1]) +
                     CV_SQR(templMean[2]) + CV_SQR(templMean[3]);//计算所有通道的均值的平方和

        if( numType != 1 )//匹配方式不是相关系数，对模板均值向量和templNorm重新赋值
        {
            templMean = Scalar::all(0);
            templNorm = templSum2;
        }
        
        templSum2 /= invArea;
        templNorm = sqrt(templNorm);
        templNorm /= sqrt(invArea); // care of accuracy here

        q0 = (double*)sqsum.data;
        q1 = q0 + templ.cols*cn;
        q2 = (double*)(sqsum.data + templ.rows*sqsum.step);
        q3 = q2 + templ.cols*cn;
    }

    //下面就是在结果图像中进行查找匹配的结果位置，代码略去，具体可参考OpenCV源代码
