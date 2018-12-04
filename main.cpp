#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <eigen3/Eigen/Eigen>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>

using std::cout;
using std::endl;
using std::cerr;

int main(int argc, char *argv[])
{
    std::string path_to_data(argv[1]);

    // Step0.0: Read in images
    int nImages = 0;
    const std::string leftVideo(path_to_data + "/CAM101.avi");
    cv::VideoCapture leftCap;
    if (leftVideo.substr(leftVideo.size() - 4) == ".avi")
    {
        leftCap.open(leftVideo);
        if (!leftCap.isOpened())
        {
            cout << "open video failed." << endl;
        }

        nImages = leftCap.get(CV_CAP_PROP_FRAME_COUNT);
    }

    for (int ni = 0; ni < nImages; ++ni)
    {
        // Step1: visualize them
        cv::Mat image;
        leftCap >> image;

        if (ni < 500)
            continue;

        // Step0.1: Read in pixels judged as lane
        std::string filepath = "/home/lightol/backup/cluster_post/";
        std::string filename = filepath + std::to_string(ni) + ".png";
        cv::Mat im = cv::imread(filename, cv::IMREAD_GRAYSCALE);

        int cols = im.cols;
        int rows = im.rows;
        int tmpID = 0;

        std::map<int, std::vector<Eigen::Vector2i> > mLanes;
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                tmpID = (int) im.at<uchar>(i, j);

                if (tmpID > 0)
                {
                    mLanes[tmpID].push_back(Eigen::Vector2i(j, i));
                }
            }
        }

        std::vector<Eigen::Vector3i> vRGBs;
        vRGBs.push_back(Eigen::Vector3i(255, 0, 0));
        vRGBs.push_back(Eigen::Vector3i(0, 255, 0));
        vRGBs.push_back(Eigen::Vector3i(0, 0, 255));
        vRGBs.push_back(Eigen::Vector3i(255, 255, 0));
        vRGBs.push_back(Eigen::Vector3i(255, 0, 255));
        vRGBs.push_back(Eigen::Vector3i(0, 255, 255));

        // Step1: 将RGB图转换为灰度图,车道线像素置为255,即纯白,其它像素置为纯黑
        cv::Mat imageGray(image.size(), CV_8UC1);
        for (int i = 0; i < imageGray.rows; ++i)
        {
            for (int j = 0; j < imageGray.cols; ++j)
            {
                imageGray.at<uchar>(i,j) = uchar(0);
            }
        }

        for (auto iter = mLanes.begin(); iter != mLanes.end(); iter++)
        {
            std::vector<Eigen::Vector2i> pts = iter->second;
            for (const Eigen::Vector2i &pt: pts)
            {
                // 只考虑图像下半部分的车道线像素,因为接近消隐点的地方像素误差过大,不予考虑
                if (pt[1] > 480)
                {
                    imageGray.at<uchar>(pt[1], pt[0]) = uchar(255);
                }
            }
        }

        // Step2: 找出图像中所有的连通域，得到描绘起边界的点集
//        cv::Mat imageBinary;
//        cv::threshold(imageGray, imageBinary, 100, 255, CV_THRESH_BINARY);
        std::vector<std::vector<cv::Point> > vContours;
        // 这个函数会自动把灰度图转化为二值图,即非零的像素(1~255)置1,零仍然保持为0
        cv::findContours(imageGray, vContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        int nC = 0;
        for (auto contour : vContours)
        {
            // 剔除面积比较小的连通域，认为是噪声
            double tmpArea = fabs(cv::contourArea(contour));
            if (tmpArea < 2500)
            {
                continue;
            }

            Eigen::Vector3i rgb = vRGBs[nC++ % 6];
            for (const cv::Point &pt: contour)
            {
                cv::circle(image, pt, 1, cv::Scalar(rgb[0], rgb[1], rgb[2]));
            }
        }

        CvMat imMsg = image;
        CvFont font;
        cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 3, 3, 0, 2);
        const char* numLane = std::to_string(nC).c_str();
        cvPutText(&imMsg, numLane, cvPoint(280,270), &font, cvScalar(0, 0, 255));

        cv::imshow("image", image);
        cv::waitKey(10);
    }
}