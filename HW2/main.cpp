#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

void showImg(const string& name, const Mat& img) {
    imshow(name, img);
    waitKey(10); // 短暂等待刷新
}
// 增强版霍夫直线检测：动态阈值 + 更强NMS + 合并接近直线
struct LineCandidate {float rho; float theta; int votes;};
vector<Vec2f> myHoughLinesEnhanced(const Mat& edge, int minVotes, float mergeRho=10.f, float mergeThetaDeg=2.f, int topK=50) {
    int W=edge.cols, H=edge.rows; int maxDist=(int)std::lround(std::sqrt(W*W+H*H));
    int rhoRange = maxDist*2 + 1; int thetaRange=180; // [0,180)
    vector<int> acc(rhoRange*thetaRange,0);
    vector<double> sinT(thetaRange), cosT(thetaRange);
    for(int t=0;t<thetaRange;t++){ double th=t*CV_PI/180.0; sinT[t]=sin(th); cosT[t]=cos(th);}    
    for(int y=0;y<H;y++){ const uchar* p=edge.ptr<uchar>(y); for(int x=0;x<W;x++) if(p[x]){
        for(int t=0;t<thetaRange;t++){ double rho=x*cosT[t]+y*sinT[t]; int r=(int)cvRound(rho)+maxDist; acc[r*thetaRange+t]++; }
    }}
    int maxVote=*max_element(acc.begin(), acc.end());
    int dynThresh = std::max(minVotes, (int)(maxVote*0.5)); // 动态阈值
    vector<LineCandidate> cand; cand.reserve(500);
    // 局部极大值 (3x3) + 阈值
    for(int r=1;r<rhoRange-1;r++){
        for(int t=1;t<thetaRange-1;t++){
            int v=acc[r*thetaRange+t]; if(v<dynThresh) continue; bool isMax=true;
            for(int dr=-1;dr<=1 && isMax;dr++) for(int dt=-1;dt<=1;dt++) if(dr||dt){ if(acc[(r+dr)*thetaRange+(t+dt)]>v){isMax=false; break;}};
            if(isMax){ cand.push_back({(float)(r-maxDist), (float)(t*CV_PI/180.f), v}); }
        }
    }
    // 根据票数排序
    sort(cand.begin(), cand.end(), [](auto& a, auto& b){return a.votes>b.votes;});
    // 合并接近的直线
    vector<Vec2f> result; vector<LineCandidate> kept;
    for(auto &c : cand){
        bool merged=false; for(auto &k: kept){ if(fabs(c.rho - k.rho) < mergeRho && fabs((c.theta - k.theta)*180.0/CV_PI) < mergeThetaDeg){ merged=true; if(c.votes>k.votes){ k=c; } break; }}
        if(!merged) kept.push_back(c);
        if((int)kept.size()>=topK) break;
    }
    for(auto &k: kept) result.push_back(Vec2f(k.rho, k.theta));
    return result;
}
// 新增：直线候选过滤（根据图像内有效段长度 + 边缘支持率）
vector<Vec2f> filterLines(const Mat& edge, const vector<Vec2f>& inLines, double minSupportRatio=0.35, int sampleStep=2, int minLength=60) {
    int W=edge.cols, H=edge.rows; vector<Vec2f> out; out.reserve(inLines.size());
    for(auto &l : inLines){ float rho=l[0], theta=l[1]; double ct=cos(theta), st=sin(theta);
        // 计算与图像边界的交点集合
        vector<Point2f> pts; pts.reserve(4);
        // x = 0 -> y
        if(fabs(st)>1e-6){ double y = (rho - 0*ct)/st; if(y>=0 && y<H) pts.emplace_back(0.f,(float)y); }
        // x = W-1 -> y
        if(fabs(st)>1e-6){ double y = (rho - (W-1)*ct)/st; if(y>=0 && y<H) pts.emplace_back((float)(W-1),(float)y); }
        // y = 0 -> x
        if(fabs(ct)>1e-6){ double x = (rho - 0*st)/ct; if(x>=0 && x<W) pts.emplace_back((float)x,0.f); }
        // y = H-1 -> x
        if(fabs(ct)>1e-6){ double x = (rho - (H-1)*st)/ct; if(x>=0 && x<W) pts.emplace_back((float)x,(float)(H-1)); }
        if(pts.size()<2) continue;
        // 选取最远两点
        Point2f p1,p2; double maxd=-1; for(size_t i=0;i<pts.size();i++) for(size_t j=i+1;j<pts.size();j++){ double d=norm(pts[i]-pts[j]); if(d>maxd){ maxd=d; p1=pts[i]; p2=pts[j]; }}
        if(maxd < minLength) continue;
        // 采样统计边缘支持
        int samples=(int)maxd/sampleStep; if(samples<10) samples=10; int support=0; for(int i=0;i<=samples;i++){ float t=i/(float)samples; float x=p1.x + (p2.x-p1.x)*t; float y=p1.y + (p2.y-p1.y)*t; int xi=cvRound(x), yi=cvRound(y); if(xi<1||xi>=W-1||yi<1||yi>=H-1) continue; bool hit=false; for(int dy=-1;dy<=1 && !hit;dy++) for(int dx=-1;dx<=1;dx++){ int xx=xi+dx, yy=yi+dy; if(xx>=0&&xx<W&&yy>=0&&yy<H && edge.at<uchar>(yy,xx)>0){ hit=true; break; }} if(hit) support++; }
        double ratio = samples>0 ? support/(double)samples : 0; if(ratio >= minSupportRatio) out.push_back(l);
    }
    return out;
}
// 增强版霍夫圆检测：按半径分层投票 + 梯度方向 + 动态阈值 + NMS + 合并
struct CircleCandidate {Point center; int r; int votes;};
vector<Vec3f> myHoughCirclesEnhanced(const Mat& edge, const Mat& gray, int minR, int maxR, double relativeThresh=0.6, int minCenterDist=20, int maxCandidatesPerRadius=20) {
    int rows=edge.rows, cols=edge.cols; Mat gx, gy; Sobel(gray, gx, CV_32F,1,0,3); Sobel(gray, gy, CV_32F,0,1,3);
    int rRange=maxR-minR+1; vector<vector<CircleCandidate>> allCands;
    // 逐半径投票
    for(int r=minR; r<=maxR; r++){
        Mat acc = Mat::zeros(rows, cols, CV_16U);
        for(int y=0;y<rows;y++){ const uchar* ep=edge.ptr<uchar>(y); for(int x=0;x<cols;x++) if(ep[x]){
            float dxv=gx.at<float>(y,x), dyv=gy.at<float>(y,x); if(fabs(dxv)<1e-5 && fabs(dyv)<1e-5) continue; float ang=atan2(dyv,dxv); float cs=cos(ang), sn=sin(ang);
            int a1=cvRound(x + r*cs), b1=cvRound(y + r*sn); if(a1>=0&&a1<cols&&b1>=0&&b1<rows) acc.at<unsigned short>(b1,a1)++;
            int a2=cvRound(x - r*cs), b2=cvRound(y + r*sn); if(a2>=0&&a2<cols&&b2>=0&&b2<rows) acc.at<unsigned short>(b2,a2)++; // 修正: 使用 a2
        }}
        double minV,maxV; Point minLoc,maxLoc; minMaxLoc(acc,&minV,&maxV,&minLoc,&maxLoc); if(maxV<=0) continue; double thr=maxV*relativeThresh;
        vector<CircleCandidate> cands;
        for(int y=1;y<rows-1;y++) for(int x=1;x<cols-1;x++){
            int v=acc.at<unsigned short>(y,x); if(v < thr) continue; bool isMax=true; for(int dy2=-1;dy2<=1 && isMax;dy2++) for(int dx2=-1;dx2<=1;dx2++) if(dx2||dy2){ if(acc.at<unsigned short>(y+dy2,x+dx2) > v){ isMax=false; break; }}
            if(isMax){ cands.push_back({Point(x,y), r, v}); }
        }
        sort(cands.begin(), cands.end(), [](auto &a, auto &b){return a.votes>b.votes;});
        if((int)cands.size() > maxCandidatesPerRadius) cands.resize(maxCandidatesPerRadius);
        allCands.push_back(cands);
    }
    // 合并所有半径候选
    vector<CircleCandidate> merged;
    for(auto &layer : allCands){ for(auto &c : layer){ bool skip=false; for(auto &m: merged){ if(norm(c.center - m.center) < minCenterDist && abs(c.r - m.r) < 5){ if(c.votes > m.votes) m = c; skip=true; break; } } if(!skip) merged.push_back(c); }}
    // 根据投票排序，输出
    sort(merged.begin(), merged.end(), [](auto &a, auto &b){return a.votes>b.votes;});
    vector<Vec3f> circles; circles.reserve(merged.size());
    for(auto &c: merged) circles.push_back(Vec3f((float)c.center.x, (float)c.center.y, (float)c.r));
    return circles;
}
// 新增：圆候选过滤（边缘命中率 + 梯度方向一致性）
vector<Vec3f> filterCircles(const Mat& edge, const Mat& gray, const vector<Vec3f>& inCircles, double edgeRatioThresh=0.5, double gradAlignThresh=0.35) {
    Mat gx, gy; Sobel(gray, gx, CV_32F,1,0,3); Sobel(gray, gy, CV_32F,0,1,3);
    int rows=edge.rows, cols=edge.cols; vector<Vec3f> out; out.reserve(inCircles.size());
    for(auto &c : inCircles){ int cx=cvRound(c[0]), cy=cvRound(c[1]), r=cvRound(c[2]); if(r<5) continue; if(cx<0||cx>=cols||cy<0||cy>=rows) continue;
        int samples = max(40, (int)cvRound(2*CV_PI*r/3)); int edgeHits=0, alignHits=0; for(int i=0;i<samples;i++){
            double ang = 2*CV_PI*i/samples; int x = cx + cvRound(r*cos(ang)); int y = cy + cvRound(r*sin(ang)); if(x<1||x>=cols-1||y<1||y>=rows-1) continue;
            // edge 检测（允许1像素膨胀）
            bool edgePixel=false; for(int dy=-1;dy<=1 && !edgePixel;dy++) for(int dx=-1;dx<=1;dx++){ int xx=x+dx, yy=y+dy; if(xx>=0&&xx<cols&&yy>=0&&yy<rows && edge.at<uchar>(yy,xx)>0){ edgePixel=true; break; }}
            if(edgePixel) edgeHits++;
            // 梯度方向与半径方向夹角
            float gxv=gx.at<float>(y,x), gyv=gy.at<float>(y,x); float mag = sqrt(gxv*gxv+gyv*gyv); if(mag>1e-3){ float rx = (x - cx)/(float)r; float ry = (y - cy)/(float)r; float dot = (gxv*rx + gyv*ry)/mag; if(fabs(dot) > 0.7f) alignHits++; }
        }
        double edgeRatio = edgeHits / (double)samples; double alignRatio = alignHits / (double)samples;
        if(edgeRatio >= edgeRatioThresh && alignRatio >= gradAlignThresh) out.push_back(c);
    }
    return out;
}

// 辅助：自适应 Canny 阈值
void adaptiveCanny(const Mat& gray, Mat& edge, double lowRatio=0.66, double highRatio=1.33) {
    vector<uchar> buf; buf.reserve(gray.total()); for(int y=0;y<gray.rows;y++){ const uchar* p=gray.ptr<uchar>(y); for(int x=0;x<gray.cols;x++) buf.push_back(p[x]); }
    nth_element(buf.begin(), buf.begin()+buf.size()/2, buf.end()); double med = buf[buf.size()/2];
    double low = std::max(0.0, lowRatio*med); double high = std::min(255.0, highRatio*med);
    Canny(gray, edge, low, high, 3);
}
// 圆检测前处理：闭运算 + 外部(一级)轮廓提取
Mat preprocessExternalCircleEdges(const Mat& edge, int closeKernelSize=5, int drawThickness=2) {
    Mat closed; morphologyEx(edge, closed, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(closeKernelSize, closeKernelSize)));
    vector<vector<Point>> contours; findContours(closed, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    Mat external = Mat::zeros(edge.size(), CV_8U);
    for (auto &c : contours) for (auto &pt : c) external.at<uchar>(pt) = 255;
    // 使用指定厚度再描绘轮廓，增强连续性
    drawContours(external, contours, -1, Scalar(255), drawThickness);
    return external;
}
// 主函数
int main() {
    string pathPrefix = "../picture/";
    {
        cout << "Processing Highway (Enhanced Line Detection)..." << endl;
        Mat src = imread(pathPrefix + "highway.png");
        if (src.empty()) {
            cerr << "Error: Cannot load highway.png" << endl;
        } else {
            Mat gray; cvtColor(src, gray, COLOR_BGR2GRAY);
            // 1) 双边滤波去除高频噪声，尽量保留边缘
            Mat grayBF; bilateralFilter(gray, grayBF, 9, 75, 75);
            // 2) 自适应 Canny 在双边后图像上
            Mat edge; adaptiveCanny(grayBF, edge);
            showImg("Highway Edge (Bilateral)", edge);
            // 3) 仅保留中下方公路区域(ROI)，不过度忽略右上角，避免误删关键信息
            Mat roadMask = Mat::zeros(src.size(), CV_8U);
            int roadY0 = cvRound(src.rows * 0.35); // 上 35% 去除
            rectangle(roadMask, Point(0, roadY0), Point(src.cols - 1, src.rows - 1), Scalar(255), FILLED);
            Mat centralMask = Mat::zeros(src.size(), CV_8U);
            int cX0 = cvRound(src.cols * 0.15), cX1 = cvRound(src.cols * 0.85);
            rectangle(centralMask, Point(cX0, roadY0), Point(cX1, src.rows - 1), Scalar(255), FILLED);
            Mat edgeRoad; bitwise_and(edge, roadMask, edgeRoad);
            Mat edgeCentral; bitwise_and(edge, centralMask, edgeCentral);
            // 4) 主检测（路面ROI）
            vector<Vec2f> lines = myHoughLinesEnhanced(edgeRoad, 70, 5.f, 1.2f, 80);
            vector<Vec2f> linesFiltered = filterLines(edgeRoad, lines, 0.40, 2, 70);
            // 5) 中央弱线二次检测（更低票阈值）
            vector<Vec2f> centralLines = myHoughLinesEnhanced(edgeCentral, 55, 6.f, 1.5f, 40);
            vector<Vec2f> centralFiltered = filterLines(edgeCentral, centralLines, 0.36, 2, 60);
            cout << "Detected lines raw: " << lines.size() << " filtered: " << linesFiltered.size() << "; central raw: " << centralLines.size() << " filtered: " << centralFiltered.size() << endl;
            // 6) 仅展示线段（在整图上绘制），颜色区分主检测/中央增强
            Mat result = src.clone();
            for (auto &ln : linesFiltered) {
                float rho = ln[0], theta = ln[1]; double a=cos(theta), b=sin(theta); double x0=a*rho, y0=b*rho;
                Point pt1(cvRound(x0 + 2000*(-b)), cvRound(y0 + 2000*(a)));
                Point pt2(cvRound(x0 - 2000*(-b)), cvRound(y0 - 2000*(a)));
                line(result, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
            }
            for (auto &ln : centralFiltered) {
                float rho = ln[0], theta = ln[1]; double a=cos(theta), b=sin(theta); double x0=a*rho, y0=b*rho;
                Point pt1(cvRound(x0 + 2000*(-b)), cvRound(y0 + 2000*(a)));
                Point pt2(cvRound(x0 - 2000*(-b)), cvRound(y0 - 2000*(a)));
                line(result, pt1, pt2, Scalar(0, 255, 255), 2, LINE_AA);
            }
            {
                Mat grayBlur; GaussianBlur(grayBF, grayBlur, Size(9,9), 2,2);
                Mat edgeCircleRaw; adaptiveCanny(grayBlur, edgeCircleRaw, 0.6, 1.4);
                bitwise_and(edgeCircleRaw, roadMask, edgeCircleRaw); // 限制到路面区域
                Mat edgeCircleClosedExternal = preprocessExternalCircleEdges(edgeCircleRaw, 7, 2);
                showImg("Highway Circles Edge External", edgeCircleClosedExternal);
                // 严苛版 Highway 圆检测参数 (提高相对阈值/支持率/方向一致性，减少候选，加强中心距离)
                vector<Vec3f> circlesRawHW = myHoughCirclesEnhanced(edgeCircleClosedExternal, grayBlur, 14, 120, 0.75, 45, 15); // minR 10->14 maxR 150->120 relativeThresh 0.50->0.75 minCenterDist 35->45 maxCandidates 30->15
                vector<Vec3f> circlesHW = filterCircles(edgeCircleClosedExternal, grayBlur, circlesRawHW, 0.60, 0.40); // edgeRatioThresh 0.48->0.60 gradAlignThresh 0.30->0.40
                cout << "Highway circles raw: " << circlesRawHW.size() << " filtered: " << circlesHW.size() << endl;
                for(auto &c : circlesHW){ Point center(cvRound(c[0]), cvRound(c[1])); int r=cvRound(c[2]); circle(result, center, r, Scalar(0,255,0), 2, LINE_AA); circle(result, center, 3, Scalar(0,128,0), -1, LINE_AA);}                
                string outDirHW = "../../result"; if(!fs::exists(outDirHW)) { try { fs::create_directories(outDirHW); } catch(...){} }
                imwrite(outDirHW + "/highway_circles_edge_external.png", edgeCircleClosedExternal);
            }
            showImg("Highway Result", result);
            // 7) 保存结果
            string outDirHW = "../../result"; if(!fs::exists(outDirHW)) { try { fs::create_directories(outDirHW); } catch(...){} }
            imwrite(outDirHW + "/highway_edge_bilateral.png", edge);
            imwrite(outDirHW + "/highway_result_lines_circles.png", result);
        }
    }
    {
        // Coin 圆检测 (调整参数 + 过滤)
        cout << "Processing Coin (Filtered Circle Detection)..." << endl;
        Mat src = imread(pathPrefix + "coin.png");
        if(src.empty()) { cerr << "Error: Cannot load coin.png" << endl; }
        else {
            Mat gray, edge; cvtColor(src, gray, COLOR_BGR2GRAY); GaussianBlur(gray, gray, Size(7,7), 1.5,1.5);
            adaptiveCanny(gray, edge, 0.5, 1.5); // 更强低阈值
            // === 圆检测预处理：闭运算 + 一级(外部)轮廓，仅在外部轮廓上检测 ===
            Mat edgeCircleRawCoin = edge.clone();
            Mat edgeCircleExternalCoin = preprocessExternalCircleEdges(edgeCircleRawCoin, 7, 2);
            showImg("Coin Edge External (Circle)", edgeCircleExternalCoin);
            // 放宽 Coin 圆检测 (降低阈值，扩大半径范围，增加候选，放宽过滤)
            vector<Vec3f> circlesRaw = myHoughCirclesEnhanced(edgeCircleExternalCoin, gray, 20, 85, 0.55, 30, 25); // minR 22->20 maxR 75->85 relativeThresh 0.55->0.45 minCenterDist 30->25 candidates 25->35
            vector<Vec3f> circles = filterCircles(edgeCircleExternalCoin, gray, circlesRaw, 0.42, 0.25); // edgeRatioThresh 0.50->0.42 gradAlignThresh 0.32->0.25
            cout << "Coins raw: " << circlesRaw.size() << " filtered: " << circles.size() << endl;
            Mat result = src.clone();
            for(auto &c: circles){ Point center(cvRound(c[0]), cvRound(c[1])); int radius=cvRound(c[2]); circle(result, center, 2, Scalar(0,255,0), -1, LINE_AA); circle(result, center, radius, Scalar(0,0,255), 2, LINE_AA);}            
            // === 新增：Coin 直线检测 ===
            Mat edgeLineCoin; adaptiveCanny(gray, edgeLineCoin, 0.65, 1.35);
            vector<Vec2f> coinLines = myHoughLinesEnhanced(edgeLineCoin, 50, 8.f, 2.f, 60);
            vector<Vec2f> coinLinesFiltered = filterLines(edgeLineCoin, coinLines, 0.42, 2, 40);
            cout << "Coin lines raw: " << coinLines.size() << " filtered: " << coinLinesFiltered.size() << endl;
            for(auto &ln : coinLinesFiltered){ float rho=ln[0], theta=ln[1]; double a=cos(theta), b=sin(theta); double x0=a*rho, y0=b*rho; Point pt1(cvRound(x0 + 2000*(-b)), cvRound(y0 + 2000*(a))); Point pt2(cvRound(x0 - 2000*(-b)), cvRound(y0 - 2000*(a))); line(result, pt1, pt2, Scalar(255,0,0), 2, LINE_AA);}            
            showImg("Coin Lines Edge", edgeLineCoin);
            Mat coinRawVis = src.clone();
            for(auto &c: circlesRaw){ Point center(cvRound(c[0]), cvRound(c[1])); int radius=cvRound(c[2]); circle(coinRawVis, center, radius, Scalar(0,128,255), 2, LINE_AA); }
            showImg("Coin Circles Filtered", result);
            // 保存圆检测预处理图
            string outDirCoin = "../../result"; if(!fs::exists(outDirCoin)) { try { fs::create_directories(outDirCoin);} catch(...){} }
            imwrite(outDirCoin + "/coin_edge.png", edge);
            imwrite(outDirCoin + "/coin_result_lines_circles.png", result);
            imwrite(outDirCoin + "/coin_circles_edge_external.png", edgeCircleExternalCoin);
        }
    }
    {
        // Seal 圆检测 (增强参数 + CLAHE + 过滤)
        cout << "Processing Seal (Filtered Circle Detection)..." << endl;
        Mat src = imread(pathPrefix + "seal.png");
        if(src.empty()) { cerr << "Error: Cannot load seal.png" << endl; }
        else {
            Mat gray, edge; cvtColor(src, gray, COLOR_BGR2GRAY);
            // 局部对比度增强
            Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8,8)); clahe->apply(gray, gray);
            GaussianBlur(gray, gray, Size(7,7), 1.5,1.5);
            adaptiveCanny(gray, edge, 0.55, 1.45);
            // === 圆检测预处理：闭运算 + 一级(外部)轮廓，仅在外部轮廓上检测 ===
            Mat edgeCircleRawSeal = edge.clone();
            Mat edgeCircleExternalSeal = preprocessExternalCircleEdges(edgeCircleRawSeal, 7, 2);
            showImg("Seal Edge External (Circle)", edgeCircleExternalSeal);
            // 放宽 Seal 圆检测 (进一步降低相对阈值，扩大半径范围，增加候选，放宽过滤)
            vector<Vec3f> circlesRaw = myHoughCirclesEnhanced(edgeCircleExternalSeal, gray, 25, 150, 0.45, 45, 35); // minR 35->30 maxR 130->150 relativeThresh 0.45->0.40 minCenterDist 45->40 candidates 35->45
            vector<Vec3f> circles = filterCircles(edgeCircleExternalSeal, gray, circlesRaw, 0.45, 0.30); // edgeRatioThresh 0.45->0.40 gradAlignThresh 0.30->0.25
            cout << "Seals raw: " << circlesRaw.size() << " filtered: " << circles.size() << endl;
            Mat result = src.clone();
            for(auto &c: circles){ Point center(cvRound(c[0]), cvRound(c[1])); int radius=cvRound(c[2]); circle(result, center, 3, Scalar(0,255,0), -1, LINE_AA); circle(result, center, radius, Scalar(255,0,0), 2, LINE_AA);}            
            // === 新增：Seal 直线检测 ===
            Mat edgeLineSeal; adaptiveCanny(gray, edgeLineSeal, 0.60, 1.40);
            vector<Vec2f> sealLines = myHoughLinesEnhanced(edgeLineSeal, 60, 8.f, 2.f, 80);
            vector<Vec2f> sealLinesFiltered = filterLines(edgeLineSeal, sealLines, 0.40, 2, 50);
            cout << "Seal lines raw: " << sealLines.size() << " filtered: " << sealLinesFiltered.size() << endl;
            for(auto &ln : sealLinesFiltered){ float rho=ln[0], theta=ln[1]; double a=cos(theta), b=sin(theta); double x0=a*rho, y0=b*rho; Point pt1(cvRound(x0 + 1500*(-b)), cvRound(y0 + 1500*(a))); Point pt2(cvRound(x0 - 1500*(-b)), cvRound(y0 - 1500*(a))); line(result, pt1, pt2, Scalar(0,0,255), 2, LINE_AA);}            
            showImg("Seal Lines Edge", edgeLineSeal);
            Mat sealRawVis = src.clone();
            for(auto &c: circlesRaw){ Point center(cvRound(c[0]), cvRound(c[1])); int radius=cvRound(c[2]); circle(sealRawVis, center, radius, Scalar(0,128,255), 2, LINE_AA); }
            showImg("Seal Circles Filtered", result);
            string outDirSeal = "../../result"; if(!fs::exists(outDirSeal)) { try { fs::create_directories(outDirSeal);} catch(...){} }
            imwrite(outDirSeal + "/seal_result_lines_circles.png", result);
            imwrite(outDirSeal + "/seal_circles_edge_raw.png", edgeCircleRawSeal);
            imwrite(outDirSeal + "/seal_circles_edge_external.png", edgeCircleExternalSeal);
        }
    }
    cout << "Press any key to exit..." << endl; waitKey(0); destroyAllWindows(); return 0;
}
