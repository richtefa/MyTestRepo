#include <iostream>

#include "selectivesearch.h"
#include <opencv/cxcore.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

struct MyStruct
{
public:
    void func() const {
        //foo_ = 5;
    }

    void func2() {
        foo_ = 5;
    }

    int foo_;
};

//void test(MyStruct const& m)
void test(MyStruct const* const m)
//void test(const MyStruct* m)
{
//    m.func();
//    m->func();
//    m->foo_ = 6;
}

int main()
{
    // load image (row-major, BGR, 8-channel unsigned)
    //cv::Mat img1 = cv::imread("/home/fabian/SampleImages/18/42206/101703.png", CV_LOAD_IMAGE_COLOR);
    //cv::Mat img1 = cv::imread("/home/fabian/SampleImages/19/84412/203407.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat img1 = cv::imread("/home/fabian/SampleImages/19/84412/203412.png", CV_LOAD_IMAGE_COLOR);
    //cv::Mat img1 = cv::imread("/home/fabian/SampleImages/20/168482/406686.png", CV_LOAD_IMAGE_COLOR);
    //cv::Mat img1 = cv::imread("/home/fabian/Downloads/SelectiveSearchCodeIJCV/000015.jpg", CV_LOAD_IMAGE_COLOR);

    assert(img1.data != NULL);
    assert(img1.type() == CV_8UC3);

    //printf("\nimg1 has size [%d x %d]", img1.rows, img1.cols);

    //cv::imshow("img1", img1);
    //cv::waitKey(0);

    // transpose (column-major)
    cv::Mat img2;
    img2 = img1.t();

    //printf("\nimg2 has size [%d x %d]", img2.rows, img2.cols);

    //cv::imshow("img2", img2);
    //cv::waitKey(0);

    // convert to float
    cv::Mat img3;
    img2.convertTo(img3, CV_32FC3, 1.0/255, 0.0);
    assert(img3.type() == CV_32FC3);

//    cv::imshow("img3", img3);
//    cv::waitKey(0);

    // convert BGR to HSV
    // @TODO: something seems to be buggy considering the boxes
    cv::Mat img4;
    //cv::cvtColor(img3, img4, cv::COLOR_BGR2HSV);
    //assert(img4.type() == CV_32FC3);
    img4 = img3;

//    cv::imshow("img4", img4);
//    cv::waitKey(0);

    // convert BGR to RGB
//    cv::Mat img4(img3.rows, img3.cols, CV_32FC3);
//    cv::Mat out[] = { img4 };
//    int from_to[] = { 0,2, 1,1, 2,0 };
//    cv::mixChannels(&img4, 1, out, 1, from_to, 3);

    // HSV channels as separate layers (opposed to the interleaved scheme of OpenCV)
    const int height = img4.rows;
    const int width = img4.cols;

    std::vector<float> data;
    data.resize(width*height*3);
    for(unsigned int y = 0; y < height; ++y)
    {
        //printf("\nprocessing row [%d]", y);

        float const* const img4_ptr = img4.ptr<float>(y);

        for(unsigned int x = 0; x < width; ++x)
        {
            for(unsigned int c = 0; c < 3; ++c)
            {
              unsigned int dst_idx = height*width*c + width*y + x;
              data[dst_idx] = img4_ptr[width*3 + x];
            }
        }
    }

    // out
    std::vector<int> rectsOut;
    std::vector<int> initSeg;
    std::vector<float> histTexOut;
    std::vector<float> histColourOut;

    // in
    std::vector<int> similarityMeasures;
    similarityMeasures.push_back(vl::SIM_COLOUR);
    similarityMeasures.push_back(vl::SIM_TEXTURE);
    similarityMeasures.push_back(vl::SIM_SIZE);
    similarityMeasures.push_back(vl::SIM_FILL);
    float threshConst = 100.0f; // i guess this is variable k in [13] mentioned in "selective search for object recognition"
    int minSize = ceil(threshConst); // default: minSize = k

    printf("\nrunning selective search");

    vl::selectivesearch(rectsOut, initSeg, histTexOut, histColourOut, &data[0], height, width, similarityMeasures, threshConst, minSize);

    int num_rects = rectsOut.size() / 4.0;

    printf("\n\nfound [%d] rects", (unsigned int)num_rects);

    std::vector<cv::Rect> myRects;

    int offset = 0;
    for(unsigned int i = 0; i < num_rects; ++i)
    {
        int tl_x = rectsOut[offset];
        int tl_y = rectsOut[offset + 1];
        int br_x = rectsOut[offset + 2];
        int br_y = rectsOut[offset + 3];
        int w = br_x - tl_x + 1;
        int h = br_y - tl_y + 1;
        assert(w > 0 && h > 0);
        assert(tl_x >= 0 && tl_x < width);
        assert(br_x >= 0 && br_x < width);
        assert(tl_y >= 0 && tl_y < height);
        assert(br_y >= 0 && br_y < height);

        //printf("\nrect[%d] at [%d,%d] has size [%d x %d]", i, tl_x, tl_y, w, h);

        if ((w >= width*0.15 && h >= height*0.15) && (w <= width*0.5 && h <= height*0.5)) {
            if (std::max(w / h, h / w) <= 1.4) {
                myRects.push_back(cv::Rect(tl_x, tl_y, w, h));
                //cv::rectangle(res, cv::Point(tl_x, tl_y), cv::Point(br_x + 1, br_y + 1), cv::Scalar(0, 1, 0), 1);
            }
        }

        offset += 4;
    }

    printf("\n\nfound [%d] size-constrained rects", (unsigned int)myRects.size());

    cv::Mat res_all = img4.clone();
    cv::Mat heatmap_raw = cv::Mat::zeros(height, width, CV_32FC1);

    for(unsigned int i = 0; i < myRects.size(); ++i)
    {
        //cv::Mat res = img4.clone();

        //cv::rectangle(res, myRects[i], cv::Scalar(0, 1, 0), 1);
        cv::rectangle(res_all, myRects[i], cv::Scalar(0, 0, 1), 1);
        heatmap_raw(myRects[i]) += 1.0f;

        //printf("\nrect[%d] at [%d,%d] has size [%d x %d]", i, myRects[i].tl().x, myRects[i].tl().y, myRects[i].width, myRects[i].height);

        //cv::imshow("res", res);
        //cv::waitKey(0);
    }

    cv::imshow("res_all", res_all);
    cv::waitKey(0);

    // get min- and max-value
    double min_val = DBL_MAX, max_val = -DBL_MAX;
    int min_idx, max_idx;
    cv::minMaxIdx(heatmap_raw, &min_val, &max_val, &min_idx, &max_idx);
    printf("\nmin-max range: [%.2f] to [%.2f]", min_val, max_val);

    cv::Mat heatmap_vis;
    cv::normalize(heatmap_raw, heatmap_vis, 0.0, 1.0, cv::NORM_MINMAX);

    cv::imshow("heatmap_vis", heatmap_vis);
    cv::waitKey(0);

    return 0;
}