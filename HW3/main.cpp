#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// 特征提取（AKAZE）
static void detectAndComputeFeatures(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) {
    Mat gray;
    if (image.channels() == 3) cvtColor(image, gray, COLOR_BGR2GRAY); else gray = image;
    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(gray, noArray(), keypoints, descriptors);
}

// Lowe 比例测试匹配（Hamming）
static void matchFeatures(const Mat& descriptors1, const Mat& descriptors2, vector<DMatch>& goodMatches, float ratio = 0.65f) {
    goodMatches.clear();
    if (descriptors1.empty() || descriptors2.empty()) return;
    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING, false);
    vector<vector<DMatch>> knnMatches;
    matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);
    for (const auto& knn : knnMatches) {
        if (knn.size() < 2) continue;
        if (knn[0].distance < ratio * knn[1].distance) goodMatches.push_back(knn[0]);
    }
}

static void filterMatches(const vector<KeyPoint>& k1, const vector<KeyPoint>& k2,
                          const vector<DMatch>& matches,
                          vector<Point2f>& pts1, vector<Point2f>& pts2) {
    pts1.clear(); pts2.clear();
    for (const auto& m : matches) {
        pts1.push_back(k1[m.queryIdx].pt);
        pts2.push_back(k2[m.trainIdx].pt);
    }
}

static bool estimateHomography(const vector<Point2f>& pts1, const vector<Point2f>& pts2, Mat& H) {
    if (pts1.size() < 4 || pts2.size() < 4) return false;
    H = findHomography(pts2, pts1, RANSAC, 4.0);
    return !H.empty();
}

// 计算透视变换后图像，输出画布大小取角点变换后的最大 x,y，边界复制
static void warpImage(const Mat& image, const Mat& H, Mat& warped) {
    vector<Point2f> corners(4);
    corners[0] = Point2f(0, 0);
    corners[1] = Point2f((float)image.cols, 0);
    corners[2] = Point2f((float)image.cols, (float)image.rows);
    corners[3] = Point2f(0, (float)image.rows);

    vector<Point2f> tc(4);
    perspectiveTransform(corners, tc, H);
    float maxX = 0.f, maxY = 0.f;
    for (int i = 0; i < 4; ++i) {
        maxX = max(maxX, tc[i].x);
        maxY = max(maxY, tc[i].y);
    }
    int W = (int)ceil(max(1.f, maxX));
    int Hout = (int)ceil(max(1.f, maxY));
    warpPerspective(image, warped, H, Size(W, Hout), INTER_LINEAR, BORDER_REPLICATE);
}

// 计算重叠区域：按左上对齐的交集（与上面的 warpImage 约定一致）
static Rect getOverlapRect(const Mat& img1, const Mat& warped2) {
    int w = min(img1.cols, warped2.cols);
    int h = min(img1.rows, warped2.rows);
    if (w <= 0 || h <= 0) return Rect();
    return Rect(0, 0, w, h);
}

// 动态规划最优拼接线 + 生成二值掩码（白：取 img1，黑：取 img2）
static void computeBlendingMask(const Mat& img1, const Mat& warped2, const Rect& overlapRect, Mat& mask) {
    int width = max(img1.cols, warped2.cols);
    int height = max(img1.rows, warped2.rows);

    // 初始化 mask：先把 img1 覆盖区域置白
    mask = Mat::zeros(Size(width, height), CV_8UC1);
    mask(Rect(0, 0, img1.cols, img1.rows)).setTo(Scalar(255));

    if (overlapRect.width <= 1 || overlapRect.height <= 1) {
        // 几乎无重叠，直接返回
        return;
    }

    // 准备重叠区的灰度图
    Mat gray1, gray2;
    cvtColor(img1(overlapRect), gray1, COLOR_BGR2GRAY);
    cvtColor(warped2(overlapRect), gray2, COLOR_BGR2GRAY);

    // 梯度（x 与 y），用幅值
    Mat gx1, gy1, gx2, gy2;
    Sobel(gray1, gx1, CV_32F, 1, 0, 3);
    Sobel(gray1, gy1, CV_32F, 0, 1, 3);
    Sobel(gray2, gx2, CV_32F, 1, 0, 3);
    Sobel(gray2, gy2, CV_32F, 0, 1, 3);
    Mat mag1, mag2;
    magnitude(gx1, gy1, mag1);
    magnitude(gx2, gy2, mag2);

    // 像素差
    Mat diff;
    absdiff(img1(overlapRect), warped2(overlapRect), diff);
    Mat diffF;
    diff.convertTo(diffF, CV_32F);
    vector<Mat> diffCh; split(diffF, diffCh);
    Mat pixelCost = diffCh[0] + diffCh[1] + diffCh[2];

    // 成本：梯度差 * 像素差
    Mat gradDiff; absdiff(mag1, mag2, gradDiff);
    Mat cost;
    multiply(gradDiff, pixelCost, cost, 1.0, CV_32F);

    // 将成本矩阵转换为 double 以避免精度/告警问题
    Mat cost64; cost.convertTo(cost64, CV_64F);

    // DP
    const int Hh = overlapRect.height;
    const int Ww = overlapRect.width;
    Mat dp(Hh, Ww, CV_64F, Scalar(DBL_MAX));
    Mat prev(Hh, Ww, CV_8S, Scalar(-1)); // 0: up, 1:left, 2:upleft

    // 顶行初始化
    for (int x = 0; x < Ww; ++x) dp.at<double>(0, x) = cost64.at<double>(0, x);

    for (int y = 1; y < Hh; ++y) {
        for (int x = 0; x < Ww; ++x) {
            double cval = cost64.at<double>(y, x);
            double best = dp.at<double>(y - 1, x);
            int from = 0; // up
            if (x > 0) {
                double left = dp.at<double>(y, x - 1);
                if (left < best) { best = left; from = 1; }
                double upleft = dp.at<double>(y - 1, x - 1);
                if (upleft < best) { best = upleft; from = 2; }
            }
            dp.at<double>(y, x) = cval + best;
            prev.at<schar>(y, x) = (schar)from;
        }
    }

    // 末行选择最小位置
    int bestX = 0; double bestVal = dp.at<double>(Hh - 1, 0);
    for (int x = 1; x < Ww; ++x) {
        double v = dp.at<double>(Hh - 1, x);
        if (v < bestVal) { bestVal = v; bestX = x; }
    }

    // 回溯得到 seam x 对每行
    vector<int> seamX(Hh, 0);
    int cy = Hh - 1, cx = bestX;
    while (cy >= 0) {
        seamX[cy] = cx;
        if (cy == 0) break;
        schar p = prev.at<schar>(cy, cx);
        if (p == 0) { // up
            cy -= 1;
        } else if (p == 1) { // left
            cx -= 1; // same y
        } else { // upleft
            cy -= 1; cx -= 1;
        }
        if (cx < 0) cx = 0;
    }

    // 按 seam 填充重叠区：左侧白(255)，右侧黑(0)
    for (int y = 0; y < Hh; ++y) {
        int sx = seamX[y];
        int yImg = overlapRect.y + y;
        int x0 = overlapRect.x;
        int x1 = overlapRect.x + Ww - 1;
        if (sx >= 0) {
            mask(Rect(x0, yImg, sx - x0 + 1, 1)).setTo(Scalar(255));
        }
        if (sx + 1 <= x1) {
            mask(Rect(sx + 1, yImg, x1 - (sx + 1) + 1, 1)).setTo(Scalar(0));
        }
    }
}

// Alpha 融合
static void blendImages(const Mat& img1, const Mat& warped2, const Mat& mask, Mat& blended) {
    int width = max(img1.cols, warped2.cols);
    int height = max(img1.rows, warped2.rows);

    Mat canvas1 = Mat::zeros(Size(width, height), CV_8UC3);
    Mat canvas2 = Mat::zeros(Size(width, height), CV_8UC3);
    img1.copyTo(canvas1(Rect(0, 0, img1.cols, img1.rows)));
    warped2.copyTo(canvas2(Rect(0, 0, warped2.cols, warped2.rows)));

    Mat alpha;
    int kx = min(151, max(3, width / 3));
    int ky = min(151, max(3, height / 3));
    if (kx % 2 == 0) ++kx; if (ky % 2 == 0) ++ky; // 保证奇数核
    blur(mask, alpha, Size(kx, ky));
    alpha.convertTo(alpha, CV_32F, 1.0f / 255.0f);

    Mat alphaInv; subtract(1.0f, alpha, alphaInv, noArray(), CV_32F);

    vector<Mat> ch1, ch2;
    split(canvas1, ch1);
    split(canvas2, ch2);
    for (int i = 0; i < 3; ++i) {
        Mat f1, f2;
        ch1[i].convertTo(f1, CV_32F);
        ch2[i].convertTo(f2, CV_32F);
        multiply(f1, alpha, f1, 1.0, CV_32F);
        multiply(f2, alphaInv, f2, 1.0, CV_32F);
        f1 += f2;
        f1.convertTo(ch1[i], CV_8U);
    }
    merge(ch1, blended);
}

static void imageStitching(const Mat& img1, const Mat& img2, Mat& blendedImage) {
    // 特征提取
    vector<KeyPoint> kp1, kp2; Mat desc1, desc2;
    detectAndComputeFeatures(img1, kp1, desc1);
    detectAndComputeFeatures(img2, kp2, desc2);

    // 特征匹配
    vector<DMatch> matches; matchFeatures(desc1, desc2, matches);
    if (matches.size() < 10) {
        cerr << "Too few matches: " << matches.size() << endl;
        blendedImage = img1.clone();
        return;
    }

    Mat visMatch; drawMatches(img1, kp1, img2, kp2, matches, visMatch);
    imshow("match", visMatch);

    // 计算单应
    vector<Point2f> p1, p2; filterMatches(kp1, kp2, matches, p1, p2);
    Mat H; if (!estimateHomography(p1, p2, H)) {
        cerr << "Homography estimation failed" << endl;
        blendedImage = img1.clone();
        return;
    }

    // 透视变换
    Mat warped2; warpImage(img2, H, warped2);
    imshow("warped image", warped2);

    // 最优拼接线 / 掩码
    Rect overlap = getOverlapRect(img1, warped2);
    Mat mask; computeBlendingMask(img1, warped2, overlap, mask);
    imshow("mask", mask);

    // 融合
    blendImages(img1, warped2, mask, blendedImage);
}

static vector<string> collectInputImagesFromArgsOrFolder(int argc, char** argv) {
    vector<string> paths;
    if (argc >= 2) {
        // argv[1..] 作为文件名（位于 resources/）
        for (int i = 1; i < argc; ++i) {
            paths.emplace_back(string("../resources/") + argv[i]);
        }
    } else {
        // 自动扫描 resources 目录
        const fs::path dir("../resources");
        if (fs::exists(dir)) {
            for (auto& p : fs::directory_iterator(dir)) {
                if (!p.is_regular_file()) continue;
                string ext = p.path().extension().string();
                transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    paths.push_back(p.path().string());
                }
            }
            sort(paths.begin(), paths.end());
        }
    }
    return paths;
}

int main(int argc, char** argv) {
    vector<string> files;
    if (argc >= 3) {
        for (int i = 1; i < argc; ++i) files.emplace_back(argv[i]);
    } else {
        // 参数不足时默认读 yosemite1-4
        files = {"yosemite1.jpg", "yosemite2.jpg", "yosemite3.jpg", "yosemite4.jpg"};
        cout << "Too few images! Use default yosemite1-4." << endl;
    }

    if (files.size() < 2) {
        cerr << "Too few images!" << endl; return -1;
    }

    Mat blendedImage = imread(string("../resources/") + files[0]);
    if (blendedImage.empty()) { cerr << "Could not open image: " << files[0] << endl; return -1; }
    for (size_t i = 1; i < files.size(); ++i) {
        Mat nextImage = imread(string("../resources/") + files[i]);
        if (nextImage.empty()) { cerr << "Could not open image: " << files[i] << endl; return -1; }
        Mat temp; imageStitching(blendedImage, nextImage, temp); blendedImage = temp;
    }

    // 可视化与保存
    int targetW = 1440;
    if (blendedImage.cols > targetW) {
        int h = (int)llround(blendedImage.rows * (targetW / (double)blendedImage.cols));
        resize(blendedImage, blendedImage, Size(targetW, h));
    }
    imshow("result", blendedImage);

    // 保存到 result/result.jpg
    try { fs::create_directories("../result"); } catch (...) {}
    string outPath = string("../result/result.jpg");
    if (!imwrite(outPath, blendedImage)) {
        cerr << "Failed to write: " << outPath << endl;
    } else {
        cout << "Saved: " << outPath << endl;
    }
    try { fs::create_directories("../../result"); } catch (...) {}
    string outPath_1 = string("../../result/result.jpg");
    if (!imwrite(outPath_1, blendedImage)) {
        cerr << "Failed to write: " << outPath_1 << endl;
    } else {
        cout << "Saved: " << outPath_1 << endl;
    }

    waitKey(0);
    return 0;
}
