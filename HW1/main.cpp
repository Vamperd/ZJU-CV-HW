#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>

using namespace cv;
using namespace std;

static std::tm localtime_safe(time_t t) {
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    tm = *std::localtime(&t);
#endif
    return tm;
}

static string nowTimeString() {
    auto now = std::chrono::system_clock::now();
    time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm = localtime_safe(tt);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

static string makeOutputFilename(bool mp4Preferred = true) {
    auto now = std::chrono::system_clock::now();
    time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm = localtime_safe(tt);
    std::ostringstream oss;
    oss << "output_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
    if (mp4Preferred) oss << ".mp4"; else oss << ".avi";
    return oss.str();
}

// 在 dst 上位置 location 叠加 overlay
static void overlayImage(const Mat &overlay, Mat &dst, Point location) {
    if (overlay.empty()) return;
    int x = std::max(location.x, 0);
    int y = std::max(location.y, 0);
    if (x >= dst.cols || y >= dst.rows) return;

    int w = std::min(overlay.cols, dst.cols - x);
    int h = std::min(overlay.rows, dst.rows - y);
    if (w <= 0 || h <= 0) return;

    Mat dstROI = dst(Rect(x, y, w, h));
        Mat ovROI = overlay(Rect(0, 0, w, h));
        ovROI.copyTo(dstROI);
}

// 底部居中字幕，带不透明背景条
static void drawSubtitle(Mat &frame, const string &text) {
    int fontFace = FONT_HERSHEY_SIMPLEX; double fontScale = 0.8; int thickness = 2;
    int baseline = 0; Size ts = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    int padY = 8; int padX = 12; int barH = ts.height + padY * 2;
    int x = (frame.cols - ts.width) / 2;
    int y = frame.rows - 20; // 距底部 20px
    Rect barRect(max(0, x - padX), max(0, y - ts.height - padY), min(frame.cols, ts.width + padX * 2), min(frame.rows, barH));
    rectangle(frame, barRect, Scalar(0, 0, 0), FILLED);
    putText(frame, text, Point(x, y), fontFace, fontScale, Scalar(255, 255, 255), thickness, LINE_AA);
}

// 右上角时间到秒
static void drawTimestamp(Mat &frame) {
    string t = nowTimeString();
    int fontFace = FONT_HERSHEY_SIMPLEX; double fontScale = 0.6; int thickness = 2;
    int baseline = 0; Size ts = getTextSize(t, fontFace, fontScale, thickness, &baseline);
    int margin = 10;
    Point org(frame.cols - ts.width - margin, margin + ts.height);
    // 半透明背景
    Rect bgRect(org.x - 6, org.y - ts.height - 4, ts.width + 12, ts.height + 8);
    bgRect &= Rect(0, 0, frame.cols, frame.rows);
    if (bgRect.area() > 0) {
        Mat roi = frame(bgRect);
        Mat overlay = roi.clone();
        rectangle(overlay, Rect(0,0,overlay.cols, overlay.rows), Scalar(0,0,0), FILLED);
        addWeighted(overlay, 0.4, roi, 0.6, 0.0, roi);
    }
    putText(frame, t, org, fontFace, fontScale, Scalar(255, 255, 255), thickness, LINE_AA);
}

// 左上角 Logo + 姓名
static void drawLogoAndName(Mat &frame, const Mat &logoOriginal, const string &name) {
    if (logoOriginal.empty()) return;
    int targetW = max(40, frame.cols / 8); // 约 12.5% 宽度
    Mat logo;
    double scale = (double)targetW / (double)logoOriginal.cols;
    int targetH = (int)std::round(logoOriginal.rows * scale);
    resize(logoOriginal, logo, Size(targetW, targetH), 0, 0, INTER_AREA);

    // 叠加 Logo
    overlayImage(logo, frame, Point(10, 10));

    // 姓名放在 Logo 底部
    int fontFace = FONT_HERSHEY_SIMPLEX; double fontScale = 0.7; int thickness = 2;
    int baseline = 0; Size ts = getTextSize(name, fontFace, fontScale, thickness, &baseline);
    int nameX = 10; int nameY = 10 + logo.rows + ts.height + 6;
    // 背景条
    Rect bgRect(nameX - 6, nameY - ts.height - 6, ts.width + 12, ts.height + 12);
    bgRect &= Rect(0, 0, frame.cols, frame.rows);
    if (bgRect.area() > 0) {
        Mat roi = frame(bgRect);
        Mat overlay = roi.clone();
        rectangle(overlay, Rect(0,0,overlay.cols, overlay.rows), Scalar(0,0,0), FILLED);
        addWeighted(overlay, 0.4, roi, 0.6, 0.0, roi);
    }
    putText(frame, name, Point(nameX, nameY), fontFace, fontScale, Scalar(255,255,255), thickness, LINE_AA);
}

// 绘制火柴人
static void drawStickman(Mat &frame, Point position, int size, Scalar color) {
    int headRadius = size / 4;
    int bodyLength = size / 2;
    int limbLength = size / 3;
    // 头部
    circle(frame, position, headRadius, color, 2, LINE_AA);
    // 身体
    Point bodyStart(position.x, position.y + headRadius);
    Point bodyEnd(position.x, position.y + headRadius + bodyLength);
    line(frame, bodyStart, bodyEnd, color, 2, LINE_AA);
    // 手臂
    Point leftArm(position.x - limbLength, position.y + headRadius + bodyLength / 3);
    Point rightArm(position.x + limbLength, position.y + headRadius + bodyLength / 3);
    line(frame, bodyStart, leftArm, color, 2, LINE_AA);
    line(frame, bodyStart, rightArm, color, 2, LINE_AA);
    // 腿
    Point leftLeg(position.x - limbLength, position.y + headRadius + bodyLength + limbLength);
    Point rightLeg(position.x + limbLength, position.y + headRadius + bodyLength + limbLength);
    line(frame, bodyEnd, leftLeg, color, 2, LINE_AA);
    line(frame, bodyEnd, rightLeg, color, 2, LINE_AA);
}

static void drawArrow(Mat &frame, Point position, Scalar color) {
    int arrowLength = 20;
    int arrowWidth = 5;
    Point tip = position;
    Point tail(position.x, position.y + arrowLength);
    line(frame, tip, tail, color, 2, LINE_AA);
    Point leftWing(tail.x - arrowWidth, tail.y - arrowWidth);
    Point rightWing(tail.x + arrowWidth, tail.y - arrowWidth);
    line(frame, tail, leftWing, color, 2, LINE_AA);
    line(frame, tail, rightWing, color, 2, LINE_AA);
}

// 生成纯算法片头：渐变背景 + 标题淡入，持续 introSeconds 秒
static void generateIntro(VideoWriter &writer, Size frameSize, double fps, const string &windowName) {
    int introSeconds = 3; // 总时长 3 秒
    int totalFrames = (int)std::round(fps * introSeconds);
    int phaseFrames = totalFrames / 2; // 每段文字显示的帧数

    vector<Point> arrows; // 存储多个箭矢的位置

    for (int i = 0; i < totalFrames; ++i) {
        double p = (double)(i % phaseFrames) / (double)max(1, phaseFrames - 1); // 0..1
        Mat frame(frameSize, CV_8UC3);

        // 动态三原色渐变背景
        for (int y = 0; y < frame.rows; ++y) {
            double ry = (double)y / (double)(frame.rows - 1);
            Vec3b c1(255, 0, 0);   // 红色
            Vec3b c2(0, 255, 0);   // 绿色
            Vec3b c3(0, 0, 255);   // 蓝色
            Vec3b c;
            for (int k = 0; k < 3; ++k) {
                c[k] = (uchar)((1.0 - ry) * c1[k] + ry * c2[k]);
                c[k] = (uchar)((1.0 - p) * c[k] + p * c3[k]); // 动态过渡到蓝色
            }
            frame.row(y).setTo(c);
        }

        // 动态光晕效果
        Point center(frame.cols / 2, frame.rows / 2);
        double maxR = hypot(frame.cols, frame.rows) / 2.0;
        double glowPhase = sin(2 * CV_PI * i / totalFrames); // 动态光晕相位
        for (int y = 0; y < frame.rows; y += 4) {
            for (int x = 0; x < frame.cols; x += 4) {
                double d = norm(Point(x, y) - center) / maxR;
                double glow = std::max(0.0, 1.0 - d) * (0.5 + 0.5 * glowPhase); // 动态光晕强度
                Vec3b &px = frame.at<Vec3b>(y, x);
                for (int k = 0; k < 3; ++k) {
                    int val = (int)std::round(px[k] + glow * 100);
                    px[k] = (uchar)std::min(255, val);
                }
            }
        }

        // 第一段和第二段文字
        string title1 = "This is a grandeur attempt to use OpenCV";
        string title2 = "The journey of CV begins from this step";
        string currentTitle = (i < phaseFrames) ? title1 : title2;

        int font = FONT_HERSHEY_SIMPLEX;
        double tScale = 0.8; // 调整字体大小
        int thick = 2;
        int bl = 0;
        Size tsize = getTextSize(currentTitle, font, tScale, thick, &bl);
        Point tOrg((frame.cols - tsize.width) / 2, frame.rows / 3); // 调整文字位置到中心偏高

        // 根据 p 设置文本颜色透明度
        double alpha = (i < phaseFrames) ? (1.0 - p) : p; // 第一段淡出，第二段渐入
        Scalar tc(255, 255, 255);

        // 文本阴影
        putText(frame, currentTitle, tOrg + Point(2, 2), font, tScale, Scalar(0, 0, 0), thick + 2, LINE_AA);
        // 用加权实现淡入/淡出
        Mat textLayer = frame.clone();
        putText(textLayer, currentTitle, tOrg, font, tScale, tc, thick, LINE_AA);
        addWeighted(textLayer, alpha, frame, 1.0 - alpha, 0.0, frame);

        // 火柴人动画
        int stickmanX = (i * frame.cols) / totalFrames; // 火柴人位置随帧数移动
        int stickmanY = frame.rows - frame.rows / 6;    // 火柴人位于底部区域
        Point stickmanPosition(stickmanX, stickmanY);
        drawStickman(frame, stickmanPosition, 30, Scalar(255, 255, 255));

        // 箭矢动画
        if (i % 10 == 0) { // 每隔 10 帧生成一个新的箭矢
            arrows.push_back(Point(stickmanX, tOrg.y + tsize.height + 20)); // 从标题下方开始
        }

        for (size_t j = 0; j < arrows.size(); ++j) {
            arrows[j].y += 10; // 箭矢向下移动
            drawArrow(frame, arrows[j], Scalar(0, 0, 255));
        }

        // 移除超出火柴人水平线的箭矢
        arrows.erase(remove_if(arrows.begin(), arrows.end(),
                               [&](Point p) { return p.y > stickmanY; }),
                     arrows.end());

        imshow(windowName, frame);
        if (writer.isOpened()) writer.write(frame);
        int key = waitKey((int)std::round(1000.0 / std::max(1.0, fps)));
        if (key == 27 || key == 'q' || key == 'Q') break;
    }
}

int main() {
    // 读取 Logo（支持带 alpha 的 PNG）
    Mat logo = imread("TRY.png", IMREAD_UNCHANGED);
    if (logo.empty()) {
        cerr <<"警告: 未找到 TRY.png,Logo 将被跳过。" << endl;
    }

    // 打开摄像头
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "无法打开摄像头。" << endl;
        return -1;
    }

    // 尺寸与帧率
    int width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    if (width <= 0 || height <= 0) { width = 1280; height = 720; }
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0 || fps > 240) fps = 30.0; // 某些摄像头返回 0

    // VideoWriter（优先 mp4，失败回退 avi）
    string windowName = "Camera";
    namedWindow(windowName, WINDOW_AUTOSIZE);

    string outName = makeOutputFilename(true);
    int fourcc = VideoWriter::fourcc('m','p','4','v');
    VideoWriter writer(outName, fourcc, fps, Size(width, height));
    if (!writer.isOpened()) {
        cerr << "mp4 打开失败，回退到 avi/XVID。" << endl;
        outName = makeOutputFilename(false);
        fourcc = VideoWriter::fourcc('X','V','I','D');
        writer.open(outName, fourcc, fps, Size(width, height));
    }
    if (!writer.isOpened()) {
        cerr << "无法创建视频文件。" << endl;
        return -2;
    }

    // 片头
    generateIntro(writer, Size(width, height), fps, windowName);

    // 主循环：显示+录制
    const string subtitle = "Failure runs through everyone's life  - Vamper";
    for (;;) {
        Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            cerr << "读取帧失败。" << endl;
            break;
        }
        if (frame.cols != width || frame.rows != height) {
            resize(frame, frame, Size(width, height));
        }

        // 覆盖右上角时间
        drawTimestamp(frame);
        // 左上角 Logo + 姓名
        drawLogoAndName(frame, logo, "Vamper");
        // 底部字幕
        drawSubtitle(frame, subtitle);

        imshow(windowName, frame);
        if (writer.isOpened()) writer.write(frame);

        int key = waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;
    }

    cap.release();
    writer.release();
    destroyAllWindows();

    cout << "已保存视频：" << outName << endl;
    return 0;
}