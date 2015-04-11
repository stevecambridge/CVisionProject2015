//Thuy-Anh Le - 260455284
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;
// Functions prototypes
struct Face_Bounding
{
    Point2d top_left;
    Point2d bottom_right;
    Mat image;
};
void Part1(images,histogramsForFaces,centers);
Mat RANSACDLT(vector<Point2d> keypoints1, vector<Point2d> keypoints2);
int* lbp_histogram(Mat img_window);
int lbp_val(Mat img, int i, int j);
bool YaleDatasetLoader(vector<Mat> &dataset, const string baseAddress, const string fileList);
Mat lbp_extract(Mat face, int W, int H);
int nearest_centre(Mat row, Mat centres);
void lbp_cluster(Mat lbp_features, vector<Face_Bounding> faces);
void lbp_main(vector<Face_Bounding> faces);

int main()
{
    // Initialize OpenCV nonfree module
    initModule_nonfree();

    /* Load the training images */
    vector<Mat> pictures;
    // put the full address of the Training Images.txt here
    const string trainingfilelistDamien = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/damien.txt";
    const string trainingfilelistSteve = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/steve.txt";
    const string trainingfilelistDan = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/dan.txt";
    const string trainingfilelistThuyanh = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/thuy-anh.txt";
    // put the full address of the Training Images folder here
    const string trainingBaseAddressDamien = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/damien";
    const string trainingBaseAddressDan = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/dan";
    const string trainingBaseAddressSteve = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/steve";
    const string trainingBaseAddressThuyanh = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/thuy-anh";
    // Load the training dataset
    YaleDatasetLoader(pictures, trainingBaseAddressDamien, trainingfilelistDamien);
    YaleDatasetLoader(pictures, trainingBaseAddressSteve, trainingfilelistSteve);
    YaleDatasetLoader(pictures, trainingBaseAddressDan, trainingfilelistDan);
    YaleDatasetLoader(pictures, trainingBaseAddressThuyanh, trainingfilelistThuyanh);

    vector<Face_Bounding> images;
    //begin damien photos
    //Close, medium and then far
    Face_Bounding face = {Point2d(117,108),Point2d(441,572),pictures[0]};
    images.push_back(face);
    face = {Point2d(61,122),Point2d(416,609),pictures[1]};
    images.push_back(face);
    face = {Point2d(53,78),Point2d(438,602),pictures[2]};
    images.push_back(face);
    face = {Point2d(65,110),Point2d(507,627),pictures[3]};
    images.push_back(face);
    face = {Point2d(71,92),Point2d(511,619),pictures[4]};
    images.push_back(face);
    face = {Point2d(189,208),Point2d(399,436),pictures[5]};
    images.push_back(face);
    face = {Point2d(192,231),Point2d(395,469),pictures[6]};
    images.push_back(face);
    face = {Point2d(154,234),Point2d(380,488),pictures[7]};
    images.push_back(face);
    face = {Point2d(176,216),Point2d(420,474),pictures[8]};
    images.push_back(face);
    face = {Point2d(142,195),Point2d(441,463),pictures[9]};
    images.push_back(face);
    face = {Point2d(236,231),Point2d(367,361),pictures[10]};
    images.push_back(face);
    face = {Point2d(217,233),Point2d(381,387),pictures[11]};
    images.push_back(face);
    face = {Point2d(189,244),Point2d(376,408),pictures[12]};
    images.push_back(face);
    face = {Point2d(218,257),Point2d(395,413),pictures[13]};
    images.push_back(face);
    face = {Point2d(215,247),Point2d(394,413),pictures[14]};
    images.push_back(face);

    //begin steve photos
    face = {Point2d(64,146),Point2d(454,659),pictures[15]};
    images.push_back(face);
    face = {Point2d(91,167),Point2d(467,684),pictures[16]};
    images.push_back(face);
    face = {Point2d(119,141),Point2d(459,650),pictures[17]};
    images.push_back(face);
    face = {Point2d(123,137),Point2d(530,639),pictures[18]};
    images.push_back(face);
    face = {Point2d(103,132),Point2d(534,668),pictures[19]};
    images.push_back(face);
    face = {Point2d(141,246),Point2d(402,558),pictures[20]};
    images.push_back(face);
    face = {Point2d(181,234),Point2d(407,539),pictures[21]};
    images.push_back(face);
    face = {Point2d(189,226),Point2d(423,531),pictures[22]};
    images.push_back(face);
    face = {Point2d(190,220),Point2d(450,530),pictures[23]};
    images.push_back(face);
    face = {Point2d(185,193),Point2d(463,528),pictures[24]};
    images.push_back(face);
    face = {Point2d(240,103),Point2d(356,262),pictures[25]};
    images.push_back(face);
    face = {Point2d(235,96),Point2d(353,278),pictures[26]};
    images.push_back(face);
    face = {Point2d(251,98),Point2d(367,250),pictures[27]};
    images.push_back(face);
    face = {Point2d(242,100),Point2d(366,266),pictures[28]};
    images.push_back(face);
    face = {Point2d(239,107),Point2d(372,263),pictures[29]};
    images.push_back(face);

    //begin dan photos
    face = {Point2d(127,11),Point2d(524,572),pictures[30]};
    images.push_back(face);
    face = {Point2d(135,121),Point2d(478,562),pictures[31]};
    images.push_back(face);
    face = {Point2d(129,123),Point2d(449,579),pictures[32]};
    images.push_back(face);
    face = {Point2d(127,117),Point2d(493,586),pictures[33]};
    images.push_back(face);
    face = {Point2d(91,116),Point2d(480,560),pictures[34]};
    images.push_back(face);
    face = {Point2d(175,178),Point2d(408,457),pictures[35]};
    images.push_back(face);
    face = {Point2d(179,187),Point2d(407,469),pictures[36]};
    images.push_back(face);
    face = {Point2d(180,190),Point2d(380,455),pictures[37]};
    images.push_back(face);
    face = {Point2d(172,190),Point2d(380,442),pictures[38]};
    images.push_back(face);
    face = {Point2d(166,190),Point2d(391,444),pictures[39]};
    images.push_back(face);
    face = {Point2d(237,228),Point2d(371,388),pictures[40]};
    images.push_back(face);
    face = {Point2d(225,234),Point2d(376,409),pictures[41]};
    images.push_back(face);
    face = {Point2d(217,231),Point2d(360,413),pictures[42]};
    images.push_back(face);
    face = {Point2d(232,217),Point2d(375,390),pictures[43]};
    images.push_back(face);
    face = {Point2d(203,227),Point2d(347,389),pictures[44]};
    images.push_back(face);

    //begin thuyanh
    face = {Point2d(127,11),Point2d(524,572),pictures[45]};
    images.push_back(face);
    face = {Point2d(135,121),Point2d(478,562),pictures[46]};
    images.push_back(face);
    face = {Point2d(129,123),Point2d(449,579),pictures[47]};
    images.push_back(face);
    face = {Point2d(127,117),Point2d(493,586),pictures[48]};
    images.push_back(face);
    face = {Point2d(91,116),Point2d(480,560),pictures[49]};
    images.push_back(face);
    face = {Point2d(175,178),Point2d(408,457),pictures[50]};
    images.push_back(face);
    face = {Point2d(179,187),Point2d(407,469),pictures[51]};
    images.push_back(face);
    face = {Point2d(180,190),Point2d(380,455),pictures[52]};
    images.push_back(face);
    face = {Point2d(172,190),Point2d(380,442),pictures[53]};
    images.push_back(face);
    face = {Point2d(166,190),Point2d(391,444),pictures[54]};
    images.push_back(face);
    face = {Point2d(237,228),Point2d(371,388),pictures[55]};
    images.push_back(face);
    face = {Point2d(225,234),Point2d(376,409),pictures[56]};
    images.push_back(face);
    face = {Point2d(217,231),Point2d(360,413),pictures[57]};
    images.push_back(face);
    face = {Point2d(232,217),Point2d(375,390),pictures[58]};
    images.push_back(face);
    face = {Point2d(203,227),Point2d(347,389),pictures[59]};
    images.push_back(face);

    //imshow("hello",images[1].image);
    //waitKey(0);

    // Call Part1 function
    Mat histogramsForFaces, centers;
    Part1(images,histogramsForFaces,centers);

    /* Load the testing images */
    vector<Mat> picturesTest;
    // put the full address of the Training Images.txt here
    const string trainingfilelistDamienTest = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/damien.txt";
    const string trainingfilelistSteveTest = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/steveTest.txt";
    const string trainingfilelistDanTest = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/danTest.txt";
    const string trainingfilelistThuyanhTest = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/thuy-anhTest.txt";
    // put the full address of the Training Images folder here
    const string trainingBaseAddressDamienTest = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/damienTest";
    const string trainingBaseAddressDanTest = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/danTest";
    const string trainingBaseAddressSteveTest = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/steveTest";
    const string trainingBaseAddressThuyanhTest = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/thuy-anhTest";
    // Load the training dataset

    YaleDatasetLoader(picturesTest, trainingBaseAddressSteveTest, trainingfilelistSteveTest);
    YaleDatasetLoader(picturesTest, trainingBaseAddressDanTest, trainingfilelistDanTest);
    YaleDatasetLoader(picturesTest, trainingBaseAddressThuyanhTest, trainingfilelistThuyanhTest);
    //YaleDatasetLoader(picturesTest, trainingBaseAddressDamienTest, trainingfilelistDamienTest);

    vector<Face_Bounding> imagesTest;
    //Steve
    Face_Bounding faceTest = {"steve",Point2d(122,106),Point2d(511,716),pictures[0]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve",Point2d(75,143),Point2d(462,661),pictures[1]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve",Point2d(138,132),Point2d(452,592),pictures[2]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve",Point2d(128,144),Point2d(436,604),pictures[3]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve",Point2d(6,68),Point2d(159,78),pictures[4]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve",Point2d(159,78),Point2d(569,591),pictures[5]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve",Point2d(71,81),Point2d(530,650),pictures[6]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve",Point2d(165,85),Point2d(518,580),pictures[7]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve",Point2d(111,74),Point2d(455,595),pictures[8]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve",Point2d(120,46),Point2d(443,515),pictures[9]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve",Point2d(100,39),Point2d(507,619),pictures[10]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve",Point2d(60,37),Point2d(426,527),pictures[11]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve",Point2d(111,56),Point2d(538,611),pictures[12]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve",Point2d(3,47),Point2d(406,548),pictures[13]};
    imagesTest.push_back(faceTest);

    //Dan pictures
    faceTest = {"dan",Point2d(86,135),Point2d(492,622),pictures[14]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan",Point2d(118,106),Point2d(520,586),pictures[15]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan",Point2d(115,144),Point2d(434,566),pictures[16]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan",Point2d(108,158),Point2d(427,587),pictures[17]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan",Point2d(95,100),Point2d(509,671),pictures[18]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan",Point2d(102,142),Point2d(490,660),pictures[19]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan",Point2d(8,158),Point2d(477,779),pictures[20]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan",Point2d(30,120),Point2d(540,739),pictures[21]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan",Point2d(17,161),Point2d(512,757),pictures[22]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan",Point2d(44,98),Point2d(580,741),pictures[23]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan",Point2d(57,135),Point2d(520,660),pictures[24]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan",Point2d(105,156),Point2d(505,625),pictures[25]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan",Point2d(56,90),Point2d(555,637),pictures[26]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan",Point2d(156,124),Point2d(576,622),pictures[27]};
    imagesTest.push_back(faceTest);

    //Thuy-Anh
    faceTest = {"thuy-anh",Point2d(119,93),Point2d(461,615),pictures[28]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh",Point2d(149,191),Point2d(480,631),pictures[29]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh",Point2d(117,194),Point2d(480,565),pictures[30]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", Point2d(115,190),Point2d(455,643),pictures[31]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", Point2d(120,100),Point2d(475,550),pictures[32]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", Point2d(142,116),Point2d(454,574),pictures[33]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", Point2d(119,132),Point2d(489,577),pictures[34]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", Point2d(144,148),Point2d(477,554),pictures[35]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", Point2d(96,138),Point2d(457,584),pictures[36]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", Point2d(182,124),Point2d(500,527),pictures[37]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh",Point2d(107,114),Point2d(413,548),pictures[38]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh",Point2d(89,139),Point2d(406,548),pictures[39]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", Point2d(111,120),Point2d(480,600),pictures[40]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", Point2d(66,118),Point2d(437,556),pictures[41]};
    imagesTest.push_back(faceTest);

    Mat histogramTestImages;
    generateHistograms(imagesTest,centers,histogramTestImages);

    return 0;
}

void generateHistograms(vector<Face_Bounding> &faces, Mat centers, Mat histogramTestImages){

    int s;
    for (s = 0; s < faces.size(); s++){
    Mat image1 = faces[s].image;
    Ptr<FeatureDetector> FeatureDetectorSIFT = FeatureDetector::create("SIFT");
    vector <KeyPoint> keyPoints1;

    //Stores the keypoints for the first and second image
    FeatureDetectorSIFT->detect(image1, keyPoints1);

    //vector of face images with bounding boxes

    Ptr<DescriptorExtractor> FeatureDescriptor = DescriptorExtractor::create("SIFT");

    //Discard the keypoints that are outside of the bounding box
    int i;
    vector<int> keyPointsInBoxIndex;
    vector<KeyPoint> keyPointsInBox;
    //cv::rectangle(image1,faces[s].top_left,faces[s].bottom_right,1,8,0);
    //point1.x =
    for (i = 0; i < keyPoints1.size(); i++){
        //if keypoint is inside then
        bool inBox = in_box(keyPoints1[i].pt,faces[s].top_left,faces[s].bottom_right);
        if (inBox){
                keyPointsInBoxIndex.push_back(i);
                keyPointsInBox.push_back(keyPoints1[i]);
        }
    }

    Mat outputImageSIFT;
    drawKeypoints(image1, keyPointsInBox, outputImageSIFT,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //imshow("Hello",outputImageSIFT);
    //waitKey(0);

    Mat extractedDescriptors1;
    //stores the SIFT descriptors for the first and second image
    FeatureDescriptor->compute(image1,keyPointsInBox,extractedDescriptors1);

    double newNorm;
    double norm = 10000;
    int histogram[50];
    int t;
    for (t=0; t < 50; t++){
        histogram[t]=0;
    }
    int center;
    vector< vector<double> > centerOfDescriptors;
    int x;
    int i;
    for(x = 0; x < extractedDescriptors1.rows; x++){
        for (i = 0; i < centers.rows; i++){
            //computes the euclidean distance between the two descriptors
            newNorm = cv::norm(extractedDescriptors1.row(x) - centers.row(i));
            //Store the best match with that descriptor
            if (newNorm < norm){
               norm = newNorm;
               center = i;
            }
        }
        //Pushes back the center to which the descriptor is matched. The centers
        //are in the same order as the descriptors
        centerOfDescriptors.push_back(centers.row(center));
        histogram[center] = histogram[center] + 1;
        norm = 10000;
        center = 0;
    }
    int p;
    for (p = 0; p < 50; p++){
    cout << histogram[p] << "\n";
    }
    cout << "hello";
    Mat histogramToMat = Mat(1,50,CV_8UC1,&histogram);
    histogramTestImages.push_back(histogramToMat);
    }
}

bool in_box(Point2d in, Point2d box1, Point2d box2)
{
    int min_x,max_x,min_y,max_y;
    min_x = min(box1.x, box2.x);
    max_x = max(box1.x, box2.x);
    min_y = min(box1.y, box2.y);
    max_y = max(box1.y, box2.y);

    if(in.x > max_x || in.x < min_x)
    {
        return false;
    }
    else if(in.y > max_y || in.y < min_y)
    {
        return false;
    }
    return true;
}

bool YaleDatasetLoader(vector<Mat> &dataset, const string baseAddress, const string fileList)
{
    ifstream infile(fileList);
    cout << "Checking fileList" << endl;
    if (!infile.is_open())
    {
        cout << "\tError: Cannot find the fileList in " << fileList << endl;
        return false;
    }
    cout << "\tOK!" << endl;

    cout << "Loading Images" << endl;
    string imgName;
    while (getline(infile, imgName))
    {
        // Parse filenames
        //unsigned first = imgName.find("B")+1;
        //unsigned second = imgName.find("P");
        //unsigned third = imgName.find("A");
        //unsigned last = imgName.find(".");
        //int subj = stoi(imgName.substr(first, second - first));
        //int pose = stoi(imgName.substr(second + 1, third - second - 1));
        //string illum = imgName.substr(third + 1, last - third - 1);
        string imgNameWithoutR = imgName.substr(0,imgName.find("\r"));

        // load image
        string imgAddress = baseAddress + '/' + imgNameWithoutR;
        //string imgAddress = "/home/thuy-anh/Assignment5/Training Images/yaleB11_P00A+000E+00.pgm";
        Mat img = imread(imgAddress, CV_LOAD_IMAGE_UNCHANGED);
        // check image data
        if (!img.data)
        {
            cout << "\tError loading image in " << imgAddress << endl;
            return false;
        }
        // convert image to double
        //img.convertTo(img, CV_64F);
        // store to dataset
        //YaleData ydata = { img, subj, pose, illum };
        dataset.push_back(img);
    }
    cout << "\tDone loading " << dataset.size() << " images" << endl;

    return true;
}

void Part1 (vector<Face_Bounding> &faces, Mat &histogramsForFaces, Mat &centers) {
    Mat descriptors;
    vector<Mat> descriptorsForEach;

    int s;
    for (s = 0; s < faces.size(); s++){
    Mat image1 = faces[s].image;
    Ptr<FeatureDetector> FeatureDetectorSIFT = FeatureDetector::create("SIFT");
    vector <KeyPoint> keyPoints1;

    //Stores the keypoints for the first and second image
    FeatureDetectorSIFT->detect(image1, keyPoints1);

    //vector of face images with bounding boxes

    Ptr<DescriptorExtractor> FeatureDescriptor = DescriptorExtractor::create("SIFT");

    //Discard the keypoints that are outside of the bounding box
    int i;
    vector<int> keyPointsInBoxIndex;
    vector<KeyPoint> keyPointsInBox;
    //cv::rectangle(image1,faces[s].top_left,faces[s].bottom_right,1,8,0);
    //point1.x =
    for (i = 0; i < keyPoints1.size(); i++){
        //if keypoint is inside then
        bool inBox = in_box(keyPoints1[i].pt,faces[s].top_left,faces[s].bottom_right);
        if (inBox){
                keyPointsInBoxIndex.push_back(i);
                keyPointsInBox.push_back(keyPoints1[i]);
        }
    }

    Mat outputImageSIFT;
    drawKeypoints(image1, keyPointsInBox, outputImageSIFT,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //imshow("Hello",outputImageSIFT);
    //waitKey(0);

    Mat extractedDescriptors1;
    //stores the SIFT descriptors for the first and second image
    FeatureDescriptor->compute(image1,keyPointsInBox,extractedDescriptors1);
    descriptors.push_back(extractedDescriptors1);
    descriptorsForEach.push_back(extractedDescriptors1);
    }
    Mat centers;
    kmeans(descriptors, 50, labels,
                TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
                   3, KMEANS_PP_CENTERS, centers);

    int j;
    for (j = 0; j < faces.size(); j++){
    //For each picture, we take its descriptors and put it in the bin with
    //the closest center
    double newNorm;
    double norm = 10000;
    int histogram[50];
    int t;
    for (t=0; t < 50; t++){
        histogram[t]=0;
        //cout << histogram[t] << "\n";
    }
    int center;
    vector< vector<double> > centerOfDescriptors;
    int x;
    int i;
    for(x = 0; x < descriptorsForEach[j].rows; x++){
        for (i = 0; i < centers.rows; i++){
            //computes the euclidean distance between the two descriptors
            newNorm = cv::norm(descriptorsForEach[j].row(x) - centers.row(i));
            //Store the best match with that descriptor
            if (newNorm < norm){
               norm = newNorm;
               center = i;
            }
        }
        //Pushes back the center to which the descriptor is matched. The centers
        //are in the same order as the descriptors
        centerOfDescriptors.push_back(centers.row(center));
        histogram[center] = histogram[center] + 1;
        norm = 10000;
        center = 0;
    }
    int p;
    for (p = 0; p < 50; p++){
    cout << histogram[p] << "\n";
    }
    cout << "hello";
    Mat histogramToMat = Mat(1,50,CV_8UC1,&histogram);
    histogramsForFaces.push_back(histogramToMat);
    }
}

void lbp_main(vector<Face_Bounding> faces)
{
    Mat descriptors;
    for(int i=0; i<faces.size(); i++)
    {
        //extract lbp features within the bounding box only and add them to the descriptors matrix
        Rect bounding_box = Rect(faces[i].top_left, faces[i].bottom_right);
        descriptors.push_back(lbp_extract(Mat(faces[i].image, bounding_box), 10, 10));
    }

    //cluster that ish
    lbp_cluster(descriptors, faces);
}


//should return lbp codebook
void lbp_cluster(Mat lbp_features, vector<Face_Bounding> faces)
{
    //clustering descriptors
    Mat labels, centres;
    kmeans(lbp_features, 50, labels,
                TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
                   3, KMEANS_PP_CENTERS, centres);

    vector< vector<int> > faces_codewords;

    //for each face
    Mat features;
    for(int i=0; i<faces.size(); i++)
    {
        Rect bounding_box = Rect(faces[i].top_left, faces[i].bottom_right);
        features = lbp_extract(Mat(faces[i].image, bounding_box), 10, 10);

        //run through and generate a histogram of code words
        vector<int> codewords;
        for(int j=0; j<50; j++)
        {
            codewords.push_back(0);
        }

        //for each feature in this face
        for(int j=0; j<features.rows; j++)
        {
            codewords[nearest_centre(features.row(j), centres)] += 1;
        }

        faces_codewords.push_back(codewords);
    }
}

//get the nearest cluster centre, ie. to which cluster does this feature belong
int nearest_centre(Mat row, Mat centres)
{
    int nearest = 0;
    int dist = norm(row, centres.row(0));
    int test_dist;
    //for each cluster centre
    for(int i=1; i<centres.rows; i++)
    {
        test_dist = norm(row, centres.row(i));
        if(test_dist < dist)
        {
            dist = test_dist;
            nearest = i;
        }
    }
    return nearest;
}

//pass the part of the picture within the bounding box
//W and H are the numbers of rows and columns of windows
//returns the matrix consisting of all extracted lbp features
Mat lbp_extract(Mat face, int W, int H)
{
    Mat window;
    int row_height = window.rows / (double)W;
    int col_width = window.cols / (double)H;
    int *histogram;
    Mat lbp_feature_row;
    Mat lbp_features;

    //loop over image and perform lbp junk on each window of face
    for(int i=0; i<W; i++)
    {
        for(int j=0; j<H; j++)
        {
            window = Mat(face, Rect(j*col_width, i*row_height, col_width, row_height));
            histogram = lbp_histogram(window);
            lbp_feature_row = Mat(1, 256, CV_8UC1, histogram);
            lbp_features.push_back(lbp_feature_row);
        }
    }

    return lbp_features;
}

//returns the histogram of lbp descriptors, given one of the W x H windows of the image
int* lbp_histogram(Mat img_window)
{
    //initialize histogram
    int *hist = new int[256];
    for(int i=0; i<256; i++)
        hist[i] = 0;

    cvtColor(img_window, img_window, CV_RGB2GRAY);
    for(int i=0; i<img_window.rows; i++)
    {
        for(int j=0; j<img_window.cols; j++)
        {
            //ignoring border pixels cos lbp would get weird
            if(i!=0 && j!=0 && i!=img_window.rows-1 && j!= img_window.cols-1)
                hist[lbp_val(img_window, i, j)] += 1;
        }
    }
    return hist;
}

//helper for lbp_histogram
int lbp_val(Mat img, int i, int j)
{
    int center = img.at<double>(i,j);
    int out = 0;
    out += (img.at<double>(i-1, j) < center);
    out += (img.at<double>(i-1, j+1) < center)*2;
    out += (img.at<double>(i, j+1) < center)*4;
    out += (img.at<double>(i+1, j+1) < center)*8;
    out += (img.at<double>(i+1, j) < center)*16;
    out += (img.at<double>(i+1, j-1) < center)*32;
    out += (img.at<double>(i, j-1) < center)*64;
    out += (img.at<double>(i-1, j-1) < center)*128;
}


//determines if the first point is within the box defined by the second two points

