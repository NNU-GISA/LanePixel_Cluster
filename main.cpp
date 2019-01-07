#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <eigen3/Eigen/Eigen>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <fstream>
#include "Solver.h"

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

        if (ni < 205)
            continue;

        // Step0.1: Read in pixels judged as lane
        std::string filepath = "/data/STCC/2018-08-01-16-28-52/cluster_post/";
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

        // Step1: 将RGB图转换为灰度图,车道线像素置为255,即纯白,其它像素置为纯黑
        cv::Mat imageGray(image.size(), CV_8UC1);
        for (int i = 0; i < imageGray.rows; ++i)
        {
            for (int j = 0; j < imageGray.cols; ++j)
            {
                imageGray.at<uchar>(i, j) = uchar(0);
            }
        }

        uchar *ptr = image.data;
        int step0 = image.step[0];
        int step1 = image.step[1];
        int channels = image.channels();
        assert(channels == 3);
        int elemSize1 = image.elemSize1();
        for (auto iter = mLanes.begin(); iter != mLanes.end(); iter++)
        {
            std::vector<Eigen::Vector2i> pts = iter->second;
            for (const Eigen::Vector2i &pt: pts)
            {
                // 只考虑图像下半部分的车道线像素,因为接近消隐点的地方像素误差过大,不予考虑
                if (pt[1] > 450)
                {
                    imageGray.at<uchar>(pt[1], pt[0]) = uchar(255);
                }

                int u = pt[0];
                int v = pt[1];
                *(ptr + v * step0 + u * step1 + 0 * elemSize1) = 255;
                *(ptr + v * step0 + u * step1 + 1 * elemSize1) = 0;
                *(ptr + v * step0 + u * step1 + 2 * elemSize1) = 0;
            }
        }

        // Step2: 找出图像中所有的连通域，得到描绘起边界的点集
        std::vector<std::vector<cv::Point> > vInitialContours;
        // 这个函数会自动把灰度图转化为二值图,即非零的像素(1~255)置1,零仍然保持为0
        cv::findContours(imageGray, vInitialContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        // Step3: 剔除面积比较小的连通域，认为是噪声
        std::vector<std::vector<cv::Point> > vContours;
        vContours.reserve(vInitialContours.size());
        for (auto contour : vInitialContours)
        {
            double tmpArea = fabs(cv::contourArea(contour));
            if (tmpArea < 2500)
            {
                continue;
            }

            int minY = contour[0].y;
            for (const cv::Point &pt : contour)
            {
                if (pt.y < minY)
                {
                    minY = pt.y;
                }
            }

            if (minY > 451)
            {
                continue;
            }

            vContours.push_back(contour);
        }

        // Step4: 对每一个连通域做直线拟合
        std::map<double, std::pair<Eigen::Vector2d, Eigen::Vector2d> > mpairEndPoints;
        for (int i = 0; i < vContours.size(); i++)
        {
            auto contour = vContours[i];
            // step4.0: 逐行扫描，求出每一行的中点
            int minY = contour[0].y;
            int maxY = contour[0].y;
            for (const cv::Point &pt : contour)
            {
                if (pt.y < minY)
                {
                    minY = pt.y;
                }
                if (pt.y > maxY)
                {
                    maxY = pt.y;
                }
            }

            std::vector<std::vector<int> > vvMedians;
            vvMedians.resize(maxY + 1);
            for (const cv::Point &pt : contour)
            {
                vvMedians[pt.y].push_back(pt.x);
            }

//            std::vector<Eigen::Vector2d> vMedians;
            std::vector<cv::Point> vMedians;
            vMedians.reserve(maxY - minY + 1);

            for (int j = minY; j < maxY; ++j)
            {
                std::vector<int> vXs = vvMedians[j];
                double median = std::accumulate(vXs.begin(), vXs.end(), 0) / vXs.size();
//                vMedians.push_back(Eigen::Vector2d(median, j));
                vMedians.push_back(cv::Point(median, j));
            }

            cv::Vec4f linePara;
            cv::fitLine(vMedians, linePara, cv::DIST_L2, 0, 1E-2, 1E-2);
            double cosTheta = linePara[0];
            double sinTheta = linePara[1];
            cv::Point Point0;
            Point0.x = linePara[2];
            Point0.y = linePara[3];

            // step4.2: 求出这条车道线的上下行边界[rowMin, rowMax]
            std::vector<double> rows;  // 表示一条车道线上的像素经过的所有row
            rows.reserve(contour.size());
            for (const cv::Point &pt : contour)
            {
                rows.push_back(pt.y);
            }
            double minv = *std::min_element(rows.begin(), rows.end());
            double maxv = *std::max_element(rows.begin(), rows.end());

            // 求出车道线的上下两个端点
            double u1 = (minv - Point0.y) * cosTheta / sinTheta + Point0.x;    // 车道线上端点对应的u
            double u2 = (maxv - Point0.y) * cosTheta / sinTheta + Point0.x;    // 车道线下端点对应的u,容易出现小于0或大于1758等极端情况
            Eigen::Vector2d p1(u1, minv);
            Eigen::Vector2d p2(u2, maxv);

            if ((p2 - p1).norm() < 250)
            {
                continue;
            }

            std::pair<Eigen::Vector2d, Eigen::Vector2d> pairEndPoints = std::make_pair(p1, p2);

            // step4.3: 求出这条车道线与图像下边界的截距
            double c = (800 - Point0.y) * cosTheta / sinTheta + Point0.x;
            mpairEndPoints[c] = pairEndPoints;
        }

        // Step4: 根据每个连通域拟合出的直线是否经过消隐点，滤除部分区域
        //        这个机制不行,在变道时车道线都不经过消隐点
        /* // 求图像中点(cols/2, rows/2)到拟合出直线的距离
        Eigen::Vector2d vp(cols / 2, rows / 2);
        auto iterTmp = mpairEndPoints.begin();
        while (iterTmp != mpairEndPoints.end())
        {
            Eigen::Vector2d p1 = iterTmp->second.first;
            Eigen::Vector2d p2 = iterTmp->second.second;
            Eigen::Vector2d p12 = (p2 - p1).normalized();
            Eigen::Vector2d p10 = vp - p1;
            Eigen::Vector3d P12 = Eigen::Vector3d::Zero();
            P12.topRows(2) = p12;
            Eigen::Vector3d P10 = Eigen::Vector3d::Zero();
            P10.topRows(2) = p10;

            double dist = P10.cross(P12).norm();

            if (dist > 50)
            {
                auto iter2 = iterTmp;
                mpairEndPoints.erase(iterTmp);
                iterTmp = iter2;
            }
            iterTmp++;
        } */

        // Step5: 对所有拟合出来的直线做一个聚类,也就是把属于同一条车道线的不同的连通域聚类在一起



        int nC = 0;
        std::vector<Eigen::Vector3i> vRGBs;
        vRGBs.push_back(Eigen::Vector3i(255, 0, 0));
        vRGBs.push_back(Eigen::Vector3i(0, 255, 0));
        vRGBs.push_back(Eigen::Vector3i(0, 0, 255));
        vRGBs.push_back(Eigen::Vector3i(255, 255, 0));
        vRGBs.push_back(Eigen::Vector3i(255, 0, 255));
        vRGBs.push_back(Eigen::Vector3i(0, 255, 255));

        cv::Mat image2 = image.clone();
        for (auto iter = mLanes.begin(); iter != mLanes.end(); iter++)
        {
            Eigen::Vector3i rgb = vRGBs[iter->first % 6];
            auto pts = iter->second;

            for (auto pt: pts)
            {
                cv::circle(image2, cv::Point(pt[0], pt[1]), 1, cv::Scalar(rgb[0], rgb[1], rgb[2]));
            }
        }

//        for (auto contour : vInitialContours)
        for (auto iter = mpairEndPoints.begin(); iter != mpairEndPoints.end(); iter++)
        {
            for (const cv::Point &pt: vContours[nC++])
            {
                cv::circle(image, pt, 1, cv::Scalar(0, 0, 255));
            }

            std::pair<Eigen::Vector2d, Eigen::Vector2d> pPts = iter->second;
            Eigen::Vector2d p1 = iter->second.first;
            Eigen::Vector2d p2 = iter->second.second;
            Eigen::Vector3i rgb = vRGBs[nC % 6];
            cv::line(image, cv::Point(p1[0], p1[1]), cv::Point(p2[0], p2[1]), cv::Scalar(rgb[0], rgb[1], rgb[2]), 4);
        }

//        CvMat imMsg = image;
//        CvFont font;
//        cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 3, 3, 0, 2);
//        const char* numLane = std::to_string(nC).c_str();
//        cvPutText(&imMsg, numLane, cvPoint(280,270), &font, cvScalar(0, 0, 255));
//
//        CvMat imMsg2 = image2;
//        CvFont font2;
//        cvInitFont(&font2, CV_FONT_HERSHEY_COMPLEX, 3, 3, 0, 2);
//        const char* numLane2 = std::to_string(ni).c_str();
//        cvPutText(&imMsg2, numLane2, cvPoint(280,270), &font2, cvScalar(0, 0, 255));

//        cv::imshow("image", image);
        cv::imshow("image", image2);
//        cv::imwrite("image0.png", image2);
//        if (nC == 3 || nC == 5)
//        {
//            cv::waitKey(1000);
//        } else
//        {
//            cv::waitKey(1);
//        }
//
//        if (nC > 5)
//        {
//            cerr << "something is wrong";
//            abort();
//        }
        char c = cv::waitKey(3);
        if (c==112)
        {
            cv::waitKey();

        } else
        {
            cv::waitKey(100);
        }

        cv::imwrite("pixel.png", image2);

    }
}