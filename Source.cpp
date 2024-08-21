#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdlib.h>
#include <string>
#include "Supp.h"

using namespace cv;
using namespace std;

void createWindowPartition(const Mat& srcI, Mat& resultWin, vector<Mat>& win, vector<Mat>& legend, int noOfImagePerCol, int noOfImagePerRow) {
    int winWidth = srcI.cols * noOfImagePerRow;
    int winHeight = srcI.rows * noOfImagePerCol;
    resultWin.create(winHeight, winWidth, srcI.type());

    win.resize(noOfImagePerCol * noOfImagePerRow);
    legend.resize(noOfImagePerCol * noOfImagePerRow);

    for (int i = 0; i < noOfImagePerCol * noOfImagePerRow; i++) {
        win[i] = resultWin(Rect((i % noOfImagePerRow) * srcI.cols, (i / noOfImagePerRow) * srcI.rows, srcI.cols, srcI.rows));
        legend[i].create(20, srcI.cols, srcI.type());
    }
}

// Function to classify shape by analyzing the contour
string classifyShape(const vector<Point>& contour) {
    vector<Point> approx;
    double epsilon = 0.043 * arcLength(contour, true);
    approxPolyDP(contour, approx, epsilon, true);

    int vertices = approx.size();
    if (vertices == 3) {
        return "Triangle";
    }
    else if (vertices == 4) {
        Rect rect = boundingRect(approx);
        double aspectRatio = (double)rect.width / rect.height;
        return (abs(aspectRatio - 1.0) <= 0.2) ? "Square" : "Rectangle";
    }
    else if (vertices == 8) {
        return "Octagon";
    }
    else {
        Moments m = moments(approx);
        double area = m.m00;
        double perimeter = arcLength(approx, true);
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);
        return (circularity > 0.8 || vertices > 8) ? "Circle" : "Unknown";
    }
}

void createWindowDisplay(const Mat& srcI, Mat* win, Mat* legend, Mat* win2, Mat* legend2) {
    srcI.copyTo(win[0]);
    srcI.copyTo(win2[0]);

    putText(legend[0], "Original", Point(5, 11), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250, 250, 250), 1);
    putText(legend[1], "Red Mask", Point(5, 11), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250, 250, 250), 1);
    putText(legend[2], "Contours", Point(5, 11), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250, 250, 250), 1);
    putText(legend[3], "Longest Contour", Point(5, 11), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250, 250, 250), 1);
    putText(legend[4], "Mask", Point(5, 11), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250, 250, 250), 1);
    putText(legend[5], "Sign Segmented", Point(5, 11), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250, 250, 250), 1);

    putText(legend2[0], "Original", Point(5, 11), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250, 250, 250), 1);
    putText(legend2[1], "Sign Segmented", Point(5, 11), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250, 250, 250), 1);
}

int main(int argc, char** argv) {
    string windowName;
    Mat threeImages[3], redMask, canvasColor, canvasGray, srcI, hsv;
    char str[256];
    Point2i center;
    vector<Scalar> colors;
    int const MAXfPt = 200;
    int t1, t2, t3, t4;
    RNG rng(0);
    String imgPattern("Inputs/Traffic signs/Red signs/*.png");
    vector<string> imageNames;

    // Generate random bright colors for drawing contours  
    for (int i = 0; i < MAXfPt; i++) {
        for (;;) {
            t1 = rng.uniform(0, 255);
            t2 = rng.uniform(0, 255);
            t3 = rng.uniform(0, 255);
            t4 = t1 + t2 + t3;
            if (t4 > 255) break;
        }
        colors.push_back(Scalar(t1, t2, t3));
    }

    // Collect all image names satisfying the image name pattern
    cv::glob(imgPattern, imageNames, true);
    for (size_t i = 0; i < imageNames.size(); ++i) {
        srcI = imread(imageNames[i]);

        if (srcI.empty()) {
            cout << "Cannot open image for reading" << endl;
            return -1;
        }

        // Resize the image to 250x250
        resize(srcI, srcI, Size(250, 250));

        // Create windows to display results
        int const noOfImagePerCol = 2, noOfImagePerRow = 3;
        Mat detailResultWin, win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
        createWindowPartition(srcI, detailResultWin, win, legend, noOfImagePerCol, noOfImagePerRow);

        int const noOfImagePerCol2 = 1, noOfImagePerRow2 = 2;
        Mat resultWin, win2[noOfImagePerRow2 * noOfImagePerCol2], legend2[noOfImagePerRow2 * noOfImagePerCol2];
        createWindowPartition(srcI, resultWin, win2, legend2, noOfImagePerCol2, noOfImagePerRow2);

        createWindowDisplay(srcI, win, legend, win2, legend2);

        // Create canvases for drawing
        canvasColor.create(srcI.rows, srcI.cols, CV_8UC3);
        canvasGray.create(srcI.rows, srcI.cols, CV_8U);
        canvasColor = Scalar(0, 0, 0);

        // Red color processing
        split(srcI, threeImages);
        redMask = (threeImages[0] * 1.5 < threeImages[2]) &
            (threeImages[1] * 1.5 < threeImages[2]);
        cvtColor(redMask, win[1], COLOR_GRAY2BGR);

        // Get contours of the red regions
        vector<vector<Point>> contours;
        findContours(redMask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        int index = -1, max = 0;

        // Find the longest contour as sign boundary
        for (int j = 0; j < contours.size(); j++) {
            if (contours[j].size() > max) {
                index = j;
                max = contours[j].size();
            }
            drawContours(canvasColor, contours, j, colors[j % MAXfPt]);
            drawContours(canvasGray, contours, j, 255);

            Moments M = moments(canvasGray);
            center.x = M.m10 / M.m00;
            center.y = M.m01 / M.m00;

            floodFill(canvasGray, center, 255);
            if (countNonZero(canvasGray) > 20) {
                sprintf_s(str, "Mask %d (area > 20)", j);
                imshow(str, canvasGray);
            }
        }
        canvasColor.copyTo(win[2]);
        if (index < 0) {
            waitKey();
            continue;
        }

        // Assume the longest contour is the correct one
        canvasGray = Scalar(0);
        drawContours(canvasGray, contours, index, 255);
        cvtColor(canvasGray, win[3], COLOR_GRAY2BGR);

        Moments M = moments(canvasGray);
        center.x = M.m10 / M.m00;
        center.y = M.m01 / M.m00;

        floodFill(canvasGray, center, 255);
        cvtColor(canvasGray, canvasGray, COLOR_GRAY2BGR);
        canvasGray.copyTo(win[4]);

        canvasColor = canvasGray & srcI;
        canvasColor.copyTo(win[5]);
        canvasColor.copyTo(win2[1]);



        // Identify and draw the approximated shape
        if (index != -1) {
            vector<Point> approx;
            approxPolyDP(Mat(contours[index]), approx, 0.043 * arcLength(contours[index], true), true);

            string shapeName = classifyShape(contours[index]);
            putText(win[2], shapeName, Point(5, 11), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250, 250, 250), 1);
            drawContours(win[2], contours, index, colors[index], 1, LINE_8, noArray(), INT_MAX, Point());

            fillPoly(canvasGray, approx, Scalar(255, 255, 255));
            canvasColor.copyTo(win[4]);
        }

        windowName = "Segmentation of " + imageNames[i] + " (detail)";
        imshow(windowName, detailResultWin);
        imshow("Traffic sign segmentation", resultWin);

        waitKey();
        destroyAllWindows();
    }

    return 0;
}


