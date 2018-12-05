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

        // Step1: 将RGB图转换为灰度图,车道线像素置为255,即纯白,其它像素置为纯黑
        cv::Mat imageGray(image.size(), CV_8UC1);
        for (int i = 0; i < imageGray.rows; ++i)
        {
            for (int j = 0; j < imageGray.cols; ++j)
            {
                imageGray.at<uchar>(i, j) = uchar(0);
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
        std::vector<std::vector<cv::Point> > vInitialContours;
        // 这个函数会自动把灰度图转化为二值图,即非零的像素(1~255)置1,零仍然保持为0
        cv::findContours(imageGray, vInitialContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        // Step3: 剔除面积比较小的连通域，认为是噪声
        std::vector<std::vector<cv::Point> > vContours;
        vContours.reserve(vInitialContours.size());
        for (auto contour : vInitialContours)
        {
            double tmpArea = fabs(cv::contourArea(contour));
            if (tmpArea > 1600)
            {
                vContours.push_back(contour);
            }
        }

        // Step4: 对每一个连通域做直线拟合
        std::map<double, std::pair<Eigen::Vector2d, Eigen::Vector2d> > mpairEndPoints;
        for (int i = 0; i < vContours.size(); i++)
        {
//            if (i != 2)
//            {
//                continue;
//            }

            auto contour = vContours[i];
            // step4.1: 求出直线的斜率与截距
            double theta = 1;
            double rho = 0;
            ceres::Problem problem;

            std::ofstream fout("pts.txt");
            for (const cv::Point &pt : contour)
            {
                fout << pt.x << " " << pt.y << endl;
                ceres::CostFunction *pCostFunction = new ceres::AutoDiffCostFunction<
                        costFunctor, 1, 1, 1>(
                        new costFunctor(pt.x, pt.y));
//                problem.AddResidualBlock(pCostFunction, new ceres::CauchyLoss(0.5), &a, &b);
                problem.AddResidualBlock(pCostFunction, nullptr, &theta, &rho);
            }
            fout.close();

//            problem.SetParameterLowerBound(&rho, 0, 0);

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 100;
//            options.minimizer_progress_to_stdout = true;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
//            cout << summary.BriefReport() << endl;
//            cout << "theta = " << theta << endl;
//            cout << "rho = " << rho << endl;
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
            double u1 = (rho - minv * sin(theta)) / cos(theta);    // 车道线上端点对应的u
            double u2 = (rho - maxv * sin(theta)) / cos(theta);    // 车道线下端点对应的u,容易出现小于0或大于1758等极端情况
            Eigen::Vector2d p1(u1, minv);
            Eigen::Vector2d p2(u2, maxv);

            if ((p2 - p1).norm() < 200)
            {
                continue;
            }

            std::pair<Eigen::Vector2d, Eigen::Vector2d> pairEndPoints = std::make_pair(p1, p2);

            // step4.3: 求出这条车道线与图像下边界的截距
            double c = (rho - 800 * sin(theta)) / cos(theta);
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
//        for (auto contour : vInitialContours)
        for (auto iter = mpairEndPoints.begin(); iter != mpairEndPoints.end(); iter++)
        {
            for (const cv::Point &pt: vContours[nC++])
            {
//                cv::circle(image, pt, 1, cv::Scalar(rgb[0], rgb[1], rgb[2]));
//                cout << pt << endl;
                cv::circle(image, pt, 1, cv::Scalar(0, 0, 255));
            }

            std::pair<Eigen::Vector2d, Eigen::Vector2d> pPts = iter->second;
            Eigen::Vector2d p1 = iter->second.first;
            Eigen::Vector2d p2 = iter->second.second;
            Eigen::Vector3i rgb = vRGBs[nC % 6];
            cv::line(image, cv::Point(p1[0], p1[1]), cv::Point(p2[0], p2[1]), cv::Scalar(rgb[0], rgb[1], rgb[2]), 4);
        }

        CvMat imMsg = image;
        CvFont font;
        cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 3, 3, 0, 2);
        const char* numLane = std::to_string(nC).c_str();
        cvPutText(&imMsg, numLane, cvPoint(280,270), &font, cvScalar(0, 0, 255));

        cv::imshow("image", image);
        cv::waitKey(1);

    }
}