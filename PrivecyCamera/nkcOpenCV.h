// OpenCV 3系, 4系 共通ヘッダーファイル
// T.Nakaguchi, CFME, Chiba Univ., 2019

#pragma once

// ヘッダーファイル
#pragma warning(disable: 4819)
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#pragma warning(default: 4819)

// バージョン取得
#define CV_VERSION_STR CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

// ビルドモード
#ifdef _DEBUG
#define CV_EXT_STR "d.lib"
#else
#define CV_EXT_STR ".lib"
#endif

// ライブラリのリンク（不要な物はコメントアウト）
#define PRE_COMPILE		0	// インストール版なら 1 個別ライブラリ使用時は 0
#define PREHEAD "opencv_"

#if PRE_COMPILE
// OpenCV3 インストール版
#pragma comment(lib, PREHEAD "world" CV_VERSION_STR CV_EXT_STR)  // 全て
#pragma comment(lib, PREHEAD "ts" CV_VERSION_STR CV_EXT_STR) // 映像関連

#else
// 個別のライブラリ指定 （CmakeやStatic版使用時など）
// 基本モジュール
#pragma comment(lib, PREHEAD "core" CV_VERSION_STR CV_EXT_STR) // Core functionality
#pragma comment(lib, PREHEAD "imgproc" CV_VERSION_STR CV_EXT_STR) // Image processing
#pragma comment(lib, PREHEAD "imgcodecs" CV_VERSION_STR CV_EXT_STR) // Image file reading and writing
#pragma comment(lib, PREHEAD "videoio" CV_VERSION_STR CV_EXT_STR) // Video I/O
#pragma comment(lib, PREHEAD "highgui" CV_VERSION_STR CV_EXT_STR) // High-level GUI
#pragma comment(lib, PREHEAD "video" CV_VERSION_STR CV_EXT_STR) // Video Analysis
//#pragma comment(lib, PREHEAD "calib3d" CV_VERSION_STR CV_EXT_STR) // Camera Calibration and 3D Reconstruction
#pragma comment(lib, PREHEAD "features2d" CV_VERSION_STR CV_EXT_STR) // 2D Features Framework
#pragma comment(lib, PREHEAD "objdetect" CV_VERSION_STR CV_EXT_STR) // Object Detection
#pragma comment(lib, PREHEAD "dnn" CV_VERSION_STR CV_EXT_STR) // Deep Neural Network module
//#pragma comment(lib, PREHEAD "ml" CV_VERSION_STR CV_EXT_STR) // Machine Learning
//#pragma comment(lib, PREHEAD "flann" CV_VERSION_STR CV_EXT_STR) // Clustering and Search in Multi-Dimensional Spaces
//#pragma comment(lib, PREHEAD "photo" CV_VERSION_STR CV_EXT_STR) // Computational Photography
//#pragma comment(lib, PREHEAD "stitching" CV_VERSION_STR CV_EXT_STR) // Images stitching
//#pragma comment(lib, PREHEAD "cudaarithm" CV_VERSION_STR CV_EXT_STR) // CUDA Operations on Matrices
//#pragma comment(lib, PREHEAD "cudabgsegm" CV_VERSION_STR CV_EXT_STR) // CUDA Background Segmentation
//#pragma comment(lib, PREHEAD "cudacodec" CV_VERSION_STR CV_EXT_STR) // CUDA Video Encoding/Decoding
//#pragma comment(lib, PREHEAD "cudafeatures2d" CV_VERSION_STR CV_EXT_STR) // CUDA Feature Detection and Description
//#pragma comment(lib, PREHEAD "cudafilters" CV_VERSION_STR CV_EXT_STR) // CUDA Image Filtering
//#pragma comment(lib, PREHEAD "cudaimgproc" CV_VERSION_STR CV_EXT_STR) // CUDA Image Processing
//#pragma comment(lib, PREHEAD "cudalegacy" CV_VERSION_STR CV_EXT_STR) // CUDA Legacy support
//#pragma comment(lib, PREHEAD "cudaobjdetect" CV_VERSION_STR CV_EXT_STR) // CUDA Object Detection
//#pragma comment(lib, PREHEAD "cudaoptflow" CV_VERSION_STR CV_EXT_STR) // CUDA Optical Flow
//#pragma comment(lib, PREHEAD "cudastereo" CV_VERSION_STR CV_EXT_STR) // CUDA Stereo Correspondence
//#pragma comment(lib, PREHEAD "cudawarping" CV_VERSION_STR CV_EXT_STR) // CUDA Image Warping
//#pragma comment(lib, PREHEAD "cudev" CV_VERSION_STR CV_EXT_STR) // CUDA Device layer
//#pragma comment(lib, PREHEAD "shape" CV_VERSION_STR CV_EXT_STR) // Shape Distance and Matching
//#pragma comment(lib, PREHEAD "superres" CV_VERSION_STR CV_EXT_STR) // Super Resolution
//#pragma comment(lib, PREHEAD "videostab" CV_VERSION_STR CV_EXT_STR) // Video Stabilization
//#pragma comment(lib, PREHEAD "viz" CV_VERSION_STR CV_EXT_STR) // 3D Visualizer

// 拡張モジュール
#pragma comment(lib, PREHEAD "aruco" CV_VERSION_STR CV_EXT_STR) // ArUco Marker Detection
//#pragma comment(lib, PREHEAD "bgsegm" CV_VERSION_STR CV_EXT_STR) // Improved Background-Foreground Segmentation Methods
//#pragma comment(lib, PREHEAD "bioinspired" CV_VERSION_STR CV_EXT_STR) // Biologically inspired vision models and derivated tools
//#pragma comment(lib, PREHEAD "ccalib" CV_VERSION_STR CV_EXT_STR) // Custom Calibration Pattern for 3D reconstruction
//#pragma comment(lib, PREHEAD "cnn_3dobj" CV_VERSION_STR CV_EXT_STR) // 3D object recognition and pose estimation API
//#pragma comment(lib, PREHEAD "cvv" CV_VERSION_STR CV_EXT_STR) // GUI for Interactive Visual Debugging of Computer Vision Programs
//#pragma comment(lib, PREHEAD "datasets" CV_VERSION_STR CV_EXT_STR) // Framework for working with different datasets
//#pragma comment(lib, PREHEAD "dnn_objdetect" CV_VERSION_STR CV_EXT_STR) // DNN used for object detection
//#pragma comment(lib, PREHEAD "dpm" CV_VERSION_STR CV_EXT_STR) // Deformable Part-based Models
//#pragma comment(lib, PREHEAD "face" CV_VERSION_STR CV_EXT_STR) // Face Analysis
//#pragma comment(lib, PREHEAD "freetype" CV_VERSION_STR CV_EXT_STR) // Drawing UTF-8 strings with freetype/harfbuzz
//#pragma comment(lib, PREHEAD "fuzzy" CV_VERSION_STR CV_EXT_STR) // Image processing based on fuzzy mathematics
//#pragma comment(lib, PREHEAD "hdf" CV_VERSION_STR CV_EXT_STR) // Hierarchical Data Format I/O routines
//#pragma comment(lib, PREHEAD "hfs" CV_VERSION_STR CV_EXT_STR) // Hierarchical Feature Selection for Efficient Image Segmentation
//#pragma comment(lib, PREHEAD "img_hash" CV_VERSION_STR CV_EXT_STR) // The module brings implementations of different image hashing algorithms.
//#pragma comment(lib, PREHEAD "line_descriptor" CV_VERSION_STR CV_EXT_STR) // Binary descriptors for lines extracted from an image
//#pragma comment(lib, PREHEAD "matlab" CV_VERSION_STR CV_EXT_STR) // MATLAB Bridge
//#pragma comment(lib, PREHEAD "optflow" CV_VERSION_STR CV_EXT_STR) // Optical Flow Algorithms
//#pragma comment(lib, PREHEAD "ovis" CV_VERSION_STR CV_EXT_STR) // OGRE 3D Visualiser
//#pragma comment(lib, PREHEAD "phase_unwrapping" CV_VERSION_STR CV_EXT_STR) // Phase Unwrapping API
//#pragma comment(lib, PREHEAD "plot" CV_VERSION_STR CV_EXT_STR) // Plot function for Mat data
//#pragma comment(lib, PREHEAD "reg" CV_VERSION_STR CV_EXT_STR) // Image Registration
//#pragma comment(lib, PREHEAD "rgbd" CV_VERSION_STR CV_EXT_STR) // RGB-Depth Processing
//#pragma comment(lib, PREHEAD "saliency" CV_VERSION_STR CV_EXT_STR) // Saliency API
//#pragma comment(lib, PREHEAD "sfm" CV_VERSION_STR CV_EXT_STR) // Structure From Motion
//#pragma comment(lib, PREHEAD "stereo" CV_VERSION_STR CV_EXT_STR) // Stereo Correspondance Algorithms
//#pragma comment(lib, PREHEAD "structured_light" CV_VERSION_STR CV_EXT_STR) // Structured Light API
//#pragma comment(lib, PREHEAD "surface_matching" CV_VERSION_STR CV_EXT_STR) // Surface Matching
//#pragma comment(lib, PREHEAD "text" CV_VERSION_STR CV_EXT_STR) // Scene Text Detection and Recognition
//#pragma comment(lib, PREHEAD "tracking" CV_VERSION_STR CV_EXT_STR) // Tracking API
//#pragma comment(lib, PREHEAD "xfeatures2d" CV_VERSION_STR CV_EXT_STR) // Extra 2D Features Framework
//#pragma comment(lib, PREHEAD "ximgproc" CV_VERSION_STR CV_EXT_STR) // Extended Image Processing
//#pragma comment(lib, PREHEAD "xobjdetect" CV_VERSION_STR CV_EXT_STR) // Extended object detection
//#pragma comment(lib, PREHEAD "xphoto" CV_VERSION_STR CV_EXT_STR) // Additional photo processing algorithms
#endif

//using namespace cv;


#include <windows.h>

namespace nkc {
	namespace ocv {

		// 待たないwaitKey()
		// 戻り値：入力キーコード，キー入力なし -1，WM_QUIT受信 0
		int noWaitKey();

		// 複数画像を縦横に連結表示
		void imShowMulti(
			cv::String winname, std::vector<cv::Mat>& imgs, // 全ての表示画像（8bit 3ch or 8bit 1chのみ)
			unsigned int cols, // 横の連結数
			unsigned int rows, // 縦の連結数
			cv::Size imgsize, // 表示する画像サイズ
			unsigned int border);

		// 縦横比を保ったリサイズ
		cv::Mat KeepAspectResize(cv::Mat& img, int width = 0, int height = 0);

		// ウインドウハンドルを取得
		HANDLE getWindowHandle(cv::String winname);

	}; // namespace ocv
}; // namespace nkc