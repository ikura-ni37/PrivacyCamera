#include "nkcOpenCV.h"
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <iomanip>
#include <chrono>
#include <Windows.h>
#include <wchar.h>
#include <cstdlib>
#include <fstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>


#define WINDOW_NAME "Privacy Camera"
#define THREAD_NUM 5
#define EVENT_NUM 8

#define EVENT_CAMERA_OUT L"event1"
#define EVENT_DETECT_IN L"event2"
#define EVENT_DETECT_OUT L"event3"
#define EVENT_BLUR_IN L"event4"
#define EVENT_BLUR_OUT_SHOW L"event5"
#define EVENT_BLUR_OUT_SAVE L"event6"
#define EVENT_SHOW_IN L"event7"
#define EVENT_SAVE_IN L"event8"

#define CROP_SIZE 300
#define WAIT_TIME 1000  //  wait for 1s
#define MIN_RECT_SIZE 50    //  min rect size for face
#define MY_COLOR cv::Scalar(240,110,100)    //  rect color

//	thread index
enum
{
    //  thread
    CAMERA = 0,
    DETECT = 1,
    BLUR = 2,
    SHOW = 3,
    SAVE = 4,
    //  blur mode
    GAUSSIAN = 5,
    NORMAL = 6,
    MEDIAN = 7
};

//  arguments struct for thread
struct ARG {

    //  shared multi thread
    std::vector<cv::Mat> frame;         
    std::vector<cv::Rect> facesRect;    

    // use for key event
    bool breakWhileLoop = false;
    bool timeStamp = true;
    bool rect = true;
    int roiPlus = 0;
    int sigma = 3;
    int blurMode = GAUSSIAN;

    cv::VideoCapture capture;
    cv::VideoWriter writer;
    cv::dnn::Net net;           
};

//  (1) thread function of getting frame
DWORD WINAPI camera(LPVOID arg);

//  (2) thread function of detecting face
DWORD WINAPI detectFace(LPVOID arg);

//  (3) thread function of bluring face
DWORD WINAPI blurFace(LPVOID arg);

//  (4) thread function of showing face
DWORD WINAPI showFrame(LPVOID arg);

//  (5) thread function of saving frame
DWORD WINAPI saveFrame(LPVOID arg);


//  function of getting time for save file name
bool getTimeStr(std::string& strTime);


int main(int argn, char* argc[])
{
    //--------------------------- setting up part ---------------------------

    //  start up option : save video file name
    std::string filename;

#ifdef _DEBUG
    //  default file name
    std::string strTime;
    if (!getTimeStr(strTime))   return -1;  /* error */
    filename = "rec" + strTime + ".mp4";
#else
    //  input save video file name
    std::cout << "input save video file name (default: 0) : ";
    std::cin >> filename;
    //  if include below character, input filename again
    //      \ / : * ? " < > |
    while (filename.find_first_of("\\/:*\?\"<>|", 0) != std::string::npos) {
        filename.clear();
        std::cout << "input save video file name (default: 0) : ";
        std::cin >> filename;
    }
    //  default file name
    if (filename == "0") {
        std::string strTime;
        if (!getTimeStr(strTime))   return -1;  /* error */
        filename = "rec" + strTime + ".mp4";
    }
    else {
        filename += ".mp4";
    }

#endif

    //  create arguments struct
    struct ARG arg;

    //  video capture
    arg.capture.open(0);
    if (!arg.capture.isOpened()) {
        std::cerr << "failed to open camera";
        return -1;  /* error */
    }

    // get video information
    const int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); /* .mp4 */
    const int width = (int)arg.capture.get(cv::CAP_PROP_FRAME_WIDTH);
    const int height = (int)arg.capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    const float fps = (float)arg.capture.get(cv::CAP_PROP_FPS);

    // create video writer
    cv::VideoWriter writer(filename, codec, fps, cv::Size(width, height));
    if (!writer.isOpened()) return -1;  /* error */
    arg.writer = writer;

    //  frame of video inicialization
    arg.frame.resize(THREAD_NUM, cv::Mat(cv::Size(width, height), CV_8UC3));

    //  dnn model(caffe) -> (2)
    const std::string caffeConfigFile = "I:\\就活データ\\PrivecyCamera\\PrivecyCamera\\deploy.prototxt";
    const std::string caffeWeightFile = "I:\\就活データ\\PrivecyCamera\\PrivecyCamera\\res10_300x300_ssd_iter_140000_fp16.caffemodel";
    arg.net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);


    //---------------------------  pipeline processing part ---------------------------

    HANDLE event[EVENT_NUM];   /* event */
    HANDLE thread[THREAD_NUM];  /* thread */
    DWORD threadID[THREAD_NUM];   /* thread ID */

    //  create events
    //      reset: automatically
    //      initial status: reset
    event[0] = CreateEvent(NULL, FALSE, FALSE, EVENT_CAMERA_OUT);
    event[1] = CreateEvent(NULL, FALSE, FALSE, EVENT_DETECT_IN);
    event[2] = CreateEvent(NULL, FALSE, FALSE, EVENT_DETECT_OUT);
    event[3] = CreateEvent(NULL, FALSE, FALSE, EVENT_BLUR_IN);
    event[4] = CreateEvent(NULL, FALSE, FALSE, EVENT_BLUR_OUT_SHOW);
    event[5] = CreateEvent(NULL, FALSE, FALSE, EVENT_BLUR_OUT_SAVE);
    event[6] = CreateEvent(NULL, FALSE, FALSE, EVENT_SHOW_IN);
    event[7] = CreateEvent(NULL, FALSE, FALSE, EVENT_SAVE_IN);

    //  create threads
    thread[CAMERA]
        = CreateThread(NULL, 0,
            camera, &arg,   /* (1) get frame from camera video */
            CREATE_SUSPENDED, &threadID[CAMERA]);  /* status: not started */
    thread[DETECT]
        = CreateThread(NULL, 0,
            detectFace, &arg,   /* (2) detect face by using dnn */
            0, &threadID[DETECT]);    /* status: started */
    thread[BLUR]
        = CreateThread(NULL, 0,
            blurFace, &arg,   /* (3) blur face */
            0, &threadID[BLUR]);    /* status: started */
    thread[SHOW]
        = CreateThread(NULL, 0,
            showFrame, &arg,   /* (4) show frame */
            0, &threadID[SHOW]);    /* status: started */
    thread[SAVE]
        = CreateThread(NULL, 0,
            saveFrame, &arg,   /* (5) save video */
            0, &threadID[SAVE]);    /* status: started */

    //  set priority
    SetThreadPriority(thread[DETECT], THREAD_PRIORITY_TIME_CRITICAL);   /* first */
    SetThreadPriority(thread[BLUR], THREAD_PRIORITY_HIGHEST);   /* second */

    //  start process
    ResumeThread(thread[CAMERA]);


    //---------------------------  ending part ---------------------------

    //  wait for all threads end processing
    WaitForMultipleObjects(THREAD_NUM, thread, TRUE, INFINITE);

    //  close hundle
    //  --- event
    for (int i = 0; i < EVENT_NUM; i++) {
        CloseHandle(event[i]);
    }
    //  --- thread
    for (int i = 0; i < THREAD_NUM; i++) {
        CloseHandle(thread[i]);
    }

    //  destroy window
    cv::destroyAllWindows();
    return 0;
}
//------------------------------------------------------------------------


// thread function of getting frame
DWORD WINAPI camera(LPVOID arg) {
    //  cast pointer and set event
    ARG* argPtr = (ARG*)arg;
    HANDLE eventOut = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_CAMERA_OUT);
    HANDLE eventNext = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_DETECT_IN);

    cv::VideoCapture capture(argPtr->capture);

    while (true) {
        //------- processing -------

        //  get frame
        if (!capture.read(argPtr->frame[CAMERA])) {
            std::cerr << "faled to get frame";
            argPtr->breakWhileLoop = true;
            return 1;
        }
        //---------------------------

        //  recieve signal of ready
        WaitForSingleObject(eventNext, WAIT_TIME);
        //  break
        if (argPtr->breakWhileLoop) break;
        //  output
        argPtr->frame[DETECT] = argPtr->frame[CAMERA].clone();
        //  notify next thread of complete
        SetEvent(eventOut);
    }
    return 0;
}


//  (2) thread function of detecting face
DWORD WINAPI detectFace(LPVOID arg) {
    //  cast pointer and set event
    ARG* argPtr = (ARG*)arg;
    HANDLE eventPrev = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_CAMERA_OUT);
    HANDLE eventIn = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_DETECT_IN);
    HANDLE eventOut = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_DETECT_OUT);
    HANDLE eventNext = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_BLUR_IN);

    //  dnn
    cv::Mat detection;
    cv::dnn::Net net(argPtr->net);

    //  catch faces
    std::vector<cv::Rect> tmpFacesRect; /* rect for save face position */
    const float confidenceThreshold = 0.5;
    float confidence;
    int x1, x2, y1, y2; /* coordinate for rect */
    const int width = (int)argPtr->capture.get(cv::CAP_PROP_FRAME_WIDTH);   /* frame width */
    const int height = (int)argPtr->capture.get(cv::CAP_PROP_FRAME_HEIGHT); /* frame height */
    int faceID = 1; /* faceID */

    while (true) {
        //  send signal of ready
        SetEvent(eventIn);
        //  recieve signal of complete
        WaitForSingleObject(eventPrev, WAIT_TIME);
        //  break
        if (argPtr->breakWhileLoop) break;

        //-------  processing -------

        //  dnn
        net.setInput(cv::dnn::blobFromImage(argPtr->frame[DETECT],
            1.0, cv::Size(CROP_SIZE, CROP_SIZE), cv::Scalar(), true));
        detection = net.forward();
        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        //  detection: 4-D matrix(NCHW)
        //      size[2]: number of detected face (unchecked)
        //      size[3]: inforamation about each face

        //  reset
        tmpFacesRect.clear();
        for (int i = 0; i < detectionMat.rows; i++) {
            confidence = detectionMat.at<float>(i, 2);
            //  compare confidence with threshold
            if (confidence > confidenceThreshold) {
                //  ROI
                x1 = static_cast<int>(detectionMat.at<float>(i, 3) * width);
                y1 = static_cast<int>(detectionMat.at<float>(i, 4) * height);
                x2 = static_cast<int>(detectionMat.at<float>(i, 5) * width);
                y2 = static_cast<int>(detectionMat.at<float>(i, 6) * height);

                //  adjustment rectangle size
                if (x2 + argPtr->roiPlus - x1 + argPtr->roiPlus > MIN_RECT_SIZE) {
                    //  x1
                    if (x1 - argPtr->roiPlus >= 0)
                        x1 -= argPtr->roiPlus;
                    else
                        x1 = 0;
                    //  x2
                    if (x2 + argPtr->roiPlus < width)
                        x2 += argPtr->roiPlus;
                    else
                        x2 = width - 1;
                }
                //  reset
                else {
                    argPtr->roiPlus = 0;
                }
                if (y2 + argPtr->roiPlus - y1 + argPtr->roiPlus > MIN_RECT_SIZE) {
                    //  y1
                    if (y1 - argPtr->roiPlus >= 0)
                        y1 -= argPtr->roiPlus;
                    else
                        y1 = 0;
                    //  y2
                    if (y2 + argPtr->roiPlus < height)
                        y2 += argPtr->roiPlus;
                    else
                        y2 = height - 1;
                }
                //  reset
                else {
                    argPtr->roiPlus = 0;
                }

                //  create rectangle
                tmpFacesRect.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));

                //  draw face information
                if (argPtr->rect) {
                    //  rectangle
                    cv::rectangle(argPtr->frame[DETECT], tmpFacesRect.back(), MY_COLOR, 2, 4);
                    //  confidence[%]
                    cv::putText(argPtr->frame[DETECT],
                        std::to_string(int(100 * confidence)) + "%",
                        cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, MY_COLOR);
                }
            }
        }

        //---------------------------

        //  recieve signal of ready
        WaitForSingleObject(eventNext, WAIT_TIME);
        //  output
        argPtr->frame[BLUR] = argPtr->frame[DETECT].clone();
        argPtr->facesRect = tmpFacesRect;
        //  notify next thread of complete
        SetEvent(eventOut);
    }
    return 0;
}


//  (3) thread function of bluring face (and draw information)
DWORD WINAPI blurFace(LPVOID arg) {
    //  cast pointer and set event
    ARG* argPtr = (ARG*)arg;
    HANDLE eventPrev = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_DETECT_OUT);
    HANDLE eventIn = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_BLUR_IN);
    HANDLE eventOut1 = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_BLUR_OUT_SHOW);
    HANDLE eventOut2 = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_BLUR_OUT_SAVE);
    HANDLE eventNext[2];
    eventNext[0] = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_SHOW_IN);
    eventNext[1] = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_SAVE_IN);

    //  use for time stamp and option key menu
    const float fps = (float)argPtr->capture.get(cv::CAP_PROP_FPS);
    int frameCount = 0;
    std::stringstream frameInfoss;
    std::string frameInfo, optionKey[2];
    optionKey[0] = "t: TimeStamp  r: Rectangle  e/s:RectangleSize  u/d: SigmaLevel";
    optionKey[1] = "Blur [g: Gaussianblur  n: Normalblur  m: Medianblur]";

    //  draw position
    cv::Point timeStampPosition(20, 20);
    cv::Point optionKeyPosition1(20, argPtr->frame[BLUR].rows - 30);
    cv::Point optionKeyPosition2(20, argPtr->frame[BLUR].rows - 15);

    while (true) {
        //  send signal of ready
        SetEvent(eventIn);
        //  recieve signal of complete
        WaitForSingleObject(eventPrev, WAIT_TIME);
        //  break
        if (argPtr->breakWhileLoop) break;

        //-------  processing -------
        switch (argPtr->blurMode) {
        case GAUSSIAN:
            for (int i = 0; i < argPtr->facesRect.size(); i++) {
                cv::GaussianBlur(argPtr->frame[BLUR](argPtr->facesRect[i]),
                    argPtr->frame[BLUR](argPtr->facesRect[i]), cv::Size(0, 0), argPtr->sigma);
            }
            break;
        case NORMAL:
            for (int i = 0; i < argPtr->facesRect.size(); i++) {
                cv::blur(argPtr->frame[BLUR](argPtr->facesRect[i]), argPtr->frame[BLUR](argPtr->facesRect[i]),
                    cv::Size(((argPtr->sigma + 1) / 2) * 2 + 1, ((argPtr->sigma + 1) / 2) * 2 + 1));
            }
            break;
        case MEDIAN:
            for (int i = 0; i < argPtr->facesRect.size(); i++) {
                cv::medianBlur(argPtr->frame[BLUR](argPtr->facesRect[i]),
                    argPtr->frame[BLUR](argPtr->facesRect[i]), ((argPtr->sigma + 1) / 2) * 2 + 1);
            }
            break;
        }

        //  draw frame information
        if (argPtr->timeStamp) {

            //  clear
            frameInfoss.str("");
            //  input to stringstream
            frameInfoss << "Frame: " << std::setfill('0') << std::right << std::setw(7) << frameCount /* frame count */
                << "  Time: " << std::fixed << std::setprecision(3) << frameCount / fps << "[s]"; /* time stamp */
            //  convert to string
            frameInfo = frameInfoss.str();
            //  draw
            cv::putText(argPtr->frame[BLUR], frameInfo, timeStampPosition,
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255));
            cv::putText(argPtr->frame[BLUR], optionKey[0], optionKeyPosition1,
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
            cv::putText(argPtr->frame[BLUR], optionKey[1], optionKeyPosition2,
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        }
        //  frame count up
        frameCount++;

        //---------------------------

        //  recieve signal of ready
        WaitForMultipleObjects(2, eventNext, TRUE, WAIT_TIME);
        //  output
        argPtr->frame[SHOW] = argPtr->frame[BLUR].clone();
        argPtr->frame[SAVE] = argPtr->frame[BLUR].clone();
        //  notify next thread of complete
        SetEvent(eventOut1);
        SetEvent(eventOut2);
    }
    return 0;
}


//  (4) thread function of showing face
DWORD WINAPI showFrame(LPVOID arg) {
    //  cast pointer and set event
    ARG* argPtr = (ARG*)arg;
    HANDLE eventPrev = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_BLUR_OUT_SHOW);
    HANDLE eventIn = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_SHOW_IN);

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    while (true) {
        //  send signal of ready
        SetEvent(eventIn);
        //  recieve signal of complete
        WaitForSingleObject(eventPrev, WAIT_TIME);
        //  break
        if (argPtr->breakWhileLoop) break;

        //-------  processing -------

        //  image show
        cv::imshow(WINDOW_NAME, argPtr->frame[SHOW]);

        //---------------------------

        /**********  key option  **********/
        const int key = cv::waitKey(1);

        //  Esc key for end
        if (key == 27) {
            argPtr->breakWhileLoop = true;
            break;
        }

        //  Stamp
        //      't' key for time stamp on/off
        else if (key == 't') {
            argPtr->timeStamp = !argPtr->timeStamp;
        }
        //      'r' key for rectangle on/off
        else if (key == 'r') {
            argPtr->rect = !argPtr->rect;
        }
        //      'e' key for expand rectangle
        else if (key == 'e') {
            argPtr->roiPlus += 5;
        }
        //      's' key for shrink rectangle
        else if (key == 's') {
            argPtr->roiPlus -= 5;
        }

        //  Blur
        //      'u' key for blur level up
        else if (key == 'u') {
            argPtr->sigma++;
        }
        //      'd' key for blur level down
        else if (key == 'd' && argPtr->sigma > 1) {
            argPtr->sigma--;
        }
        //      'g' key for gaussian blur
        else if (key == 'g') {
            argPtr->blurMode = GAUSSIAN;
        }
        //      'n' key for normal blur
        else if (key == 'n') {
            argPtr->blurMode = NORMAL;
        }
        //      'm' key for median blur
        else if (key == 'm') {
            argPtr->blurMode = MEDIAN;
        }

        /**********************************/
    }
    return 0;
}


//  (5) thread function of saving frame
DWORD WINAPI saveFrame(LPVOID arg) {
    //  cast pointer and set event
    ARG* argPtr = (ARG*)arg;
    HANDLE eventPrev = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_BLUR_OUT_SAVE);
    HANDLE eventIn = OpenEvent(EVENT_ALL_ACCESS, FALSE, EVENT_SAVE_IN);

    while (true) {
        //  send signal of ready
        SetEvent(eventIn);
        //  recieve signal of complete
        WaitForSingleObject(eventPrev, WAIT_TIME);
        //  break
        if (argPtr->breakWhileLoop) break;

        //-------  processing -------
        argPtr->writer.write(argPtr->frame[SAVE]);
        //---------------------------
    }
    return 0;
}


//  function of getting time for save file name
bool getTimeStr(std::string& strTime) {
    time_t t = time(nullptr);
    struct tm localTime;
    errno_t error = localtime_s(&localTime, &t);
    if (error != 0) return false;   /* error */
    std::stringstream ssTime;
    ssTime << "20" << localTime.tm_year - 100;
    ssTime << std::setw(2) << std::setfill('0') << localTime.tm_mon + 1;
    ssTime << std::setw(2) << std::setfill('0') << localTime.tm_mday;
    ssTime << "_";
    ssTime << std::setw(2) << std::setfill('0') << localTime.tm_hour;
    ssTime << std::setw(2) << std::setfill('0') << localTime.tm_min;
    ssTime << std::setw(2) << std::setfill('0') << localTime.tm_sec;

    strTime = ssTime.str();
    return true;
}
