//Thuy-Anh Le - 260455284
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;
// Functions prototypes
struct Face_Bounding
{
    string name;
    string pose;
    Point2d top_left;
    Point2d bottom_right;
    Mat image;
};
void Part1 (vector<Face_Bounding> &faces, vector<vector<int>> &histogramsForFaces, Mat &centers);
Mat RANSACDLT(vector<Point2d> keypoints1, vector<Point2d> keypoints2);
void generateHistograms(vector<Face_Bounding> &faces, Mat centers, Mat histogramTestImages);
vector<int> lbp_histogram(Mat img_window);
int lbp_val(Mat img, int i, int j);
bool YaleDatasetLoader(vector<Mat> &dataset, const string baseAddress, const string fileList);
Mat lbp_extract(Mat face, int W, int H);
int nearest_centre(Mat row, Mat centres);
bool in_box(Point2d in, Point2d box1, Point2d box2);
vector< vector<int> > lbp_cluster(Mat lbp_features, vector<Face_Bounding> faces, Mat centres);
vector< vector<int> > lbp_main(vector<Face_Bounding> faces, vector<Face_Bounding> test_faces);
int nearest_face(vector<int> test_hist, vector< vector<int> > trained_hists);
bool in_box(Point2d in, Point2d box1, Point2d box2);
void generateHistograms(vector<Face_Bounding> &faces, Mat &centers, vector<vector<int>> &histogramTestImages);
vector< vector<Point> > detectAndDisplay(Mat frame, CascadeClassifier face_cascade);
vector< vector<Point> > face_detect_main(Mat frame);
void face_tagging_results(Mat group, vector< vector<Point> > tagged_faces, vector<Face_Bounding> faces, Mat centres, vector< vector<int> > faces_as_codewords);
void insertInConfusionMatrix(string poseTest, string poseResult, Mat &confusionMatrix);
void faceTagging();
void detectAndDisplay(Mat frame);
void normalizeMatrices(Mat confusionMatrixDamien, Mat confusionMatrixDan, Mat confusionMatrixSteve, Mat confusionMatrixTA);

int main()
{
    // Initialize OpenCV nonfree module
    initModule_nonfree();

    /* Load the training images */
    vector<Mat> pictures;
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
    Face_Bounding face = {"damien","0",Point2d(117,108),Point2d(441,572),pictures[0]};
    images.push_back(face);
    face = {"damien","30", Point2d(61,122),Point2d(416,609),pictures[1]};
    images.push_back(face);
    face = {"damien","45", Point2d(53,78),Point2d(438,602),pictures[2]};
    images.push_back(face);
    face = {"damien","-30",Point2d(65,110),Point2d(507,627),pictures[3]};
    images.push_back(face);
    face = {"damien","-45",Point2d(71,92),Point2d(511,619),pictures[4]};
    images.push_back(face);
    face = {"damien","0", Point2d(189,208),Point2d(399,436),pictures[5]};
    images.push_back(face);
    face = {"damien","30", Point2d(192,231),Point2d(395,469),pictures[6]};
    images.push_back(face);
    face = {"damien","45",Point2d(154,234),Point2d(380,488),pictures[7]};
    images.push_back(face);
    face = {"damien","-30",Point2d(176,216),Point2d(420,474),pictures[8]};
    images.push_back(face);
    face = {"damien","-45",Point2d(142,195),Point2d(441,463),pictures[9]};
    images.push_back(face);
    face = {"damien","0",Point2d(236,231),Point2d(367,361),pictures[10]};
    images.push_back(face);
    face = {"damien","30", Point2d(217,233),Point2d(381,387),pictures[11]};
    images.push_back(face);
    face = {"damien","45",Point2d(189,244),Point2d(376,408),pictures[12]};
    images.push_back(face);
    face = {"damien","-30",Point2d(218,257),Point2d(395,413),pictures[13]};
    images.push_back(face);
    face = {"damien","-45",Point2d(215,247),Point2d(394,413),pictures[14]};
    images.push_back(face);

    //begin steve photos
    face = {"steve","45",Point2d(64,146),Point2d(454,659),pictures[15]};
    images.push_back(face);
    face = {"steve","30", Point2d(91,167),Point2d(467,684),pictures[16]};
    images.push_back(face);
    face = {"steve","0",Point2d(119,141),Point2d(459,650),pictures[17]};
    images.push_back(face);
    face = {"steve","-30",Point2d(123,137),Point2d(530,639),pictures[18]};
    images.push_back(face);
    face = {"steve","-45", Point2d(103,132),Point2d(534,668),pictures[19]};
    images.push_back(face);
    face = {"steve", "45",Point2d(141,246),Point2d(402,558),pictures[20]};
    images.push_back(face);
    face = {"steve","30", Point2d(181,234),Point2d(407,539),pictures[21]};
    images.push_back(face);
    face = {"steve", "0",Point2d(189,226),Point2d(423,531),pictures[22]};
    images.push_back(face);
    face = {"steve", "-30",Point2d(190,220),Point2d(450,530),pictures[23]};
    images.push_back(face);
    face = {"steve", "-45",Point2d(185,193),Point2d(463,528),pictures[24]};
    images.push_back(face);
    face = {"steve", "0",Point2d(240,103),Point2d(356,262),pictures[25]};
    images.push_back(face);
    face = {"steve", "45",Point2d(235,96),Point2d(353,278),pictures[26]};
    images.push_back(face);
    face = {"steve", "30",Point2d(251,98),Point2d(367,250),pictures[27]};
    images.push_back(face);
    face = {"steve", "-30",Point2d(242,100),Point2d(366,266),pictures[28]};
    images.push_back(face);
    face = {"steve", "-45",Point2d(239,107),Point2d(372,263),pictures[29]};
    images.push_back(face);

    //begin dan photos
    face = {"dan", "45", Point2d(127,11),Point2d(524,572),pictures[30]};
    images.push_back(face);
    face = {"dan", "30",Point2d(135,121),Point2d(478,562),pictures[31]};
    images.push_back(face);
    face = {"dan", "0",Point2d(129,123),Point2d(449,579),pictures[32]};
    images.push_back(face);
    face = {"dan", "-30",Point2d(127,117),Point2d(493,586),pictures[33]};
    images.push_back(face);
    face = {"dan", "-45",Point2d(91,116),Point2d(480,560),pictures[34]};
    images.push_back(face);
    face = {"dan", "45",Point2d(175,178),Point2d(408,457),pictures[35]};
    images.push_back(face);
    face = {"dan", "30",Point2d(179,187),Point2d(407,469),pictures[36]};
    images.push_back(face);
    face = {"dan", "0",Point2d(180,190),Point2d(380,455),pictures[37]};
    images.push_back(face);
    face = {"dan", "-30",Point2d(172,190),Point2d(380,442),pictures[38]};
    images.push_back(face);
    face = {"dan", "-45",Point2d(166,190),Point2d(391,444),pictures[39]};
    images.push_back(face);
    face = {"dan", "0",Point2d(237,228),Point2d(371,388),pictures[40]};
    images.push_back(face);
    face = {"dan", "45",Point2d(225,234),Point2d(376,409),pictures[41]};
    images.push_back(face);
    face = {"dan", "30",Point2d(217,231),Point2d(360,413),pictures[42]};
    images.push_back(face);
    face = {"dan", "-30",Point2d(232,217),Point2d(375,390),pictures[43]};
    images.push_back(face);
    face = {"dan", "-45",Point2d(203,227),Point2d(347,389),pictures[44]};
    images.push_back(face);

    //begin thuyanh
    face = {"thuy-anh", "0", Point2d(127,11),Point2d(524,572),pictures[45]};
      images.push_back(face);
      face = {"thuy-anh", "45", Point2d(135,121),Point2d(478,562),pictures[46]};
      images.push_back(face);
      face = {"thuy-anh", "30", Point2d(129,123),Point2d(449,579),pictures[47]};
      images.push_back(face);
      face = {"thuy-anh", "-30", Point2d(127,117),Point2d(493,586),pictures[48]};
      images.push_back(face);
      face = {"thuy-anh", "-45", Point2d(91,116),Point2d(480,560),pictures[49]};
      images.push_back(face);
      face = {"thuy-anh", "0", Point2d(175,178),Point2d(408,457),pictures[50]};
      images.push_back(face);
      face = {"thuy-anh", "45", Point2d(179,187),Point2d(407,469),pictures[51]};
      images.push_back(face);
      face = {"thuy-anh", "30", Point2d(180,190),Point2d(380,455),pictures[52]};
      images.push_back(face);
      face = {"thuy-anh", "-30", Point2d(172,190),Point2d(380,442),pictures[53]};
      images.push_back(face);
      face = {"thuy-anh", "-45", Point2d(166,190),Point2d(391,444),pictures[54]};
      images.push_back(face);
      face = {"thuy-anh", "0", Point2d(237,228),Point2d(371,388),pictures[55]};
      images.push_back(face);
      face = {"thuy-anh", "45", Point2d(225,234),Point2d(376,409),pictures[56]};
      images.push_back(face);
      face = {"thuy-anh", "30", Point2d(217,231),Point2d(360,413),pictures[57]};
      images.push_back(face);
      face = {"thuy-anh", "-30", Point2d(232,217),Point2d(375,390),pictures[58]};
      images.push_back(face);
      face = {"thuy-anh", "-45", Point2d(203,227),Point2d(347,389),pictures[59]};
      images.push_back(face);

    // Call Part1 function
    Mat centers;
    vector<vector<int>> histogramsForFaces;
    Part1(images,histogramsForFaces,centers);

    /* Load the testing images */
    vector<Mat> picturesTest;
    // put the full address of the Training Images.txt here
    const string trainingfilelistDamienTest = "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/damienTest.txt";
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
    YaleDatasetLoader(picturesTest, trainingBaseAddressDamienTest, trainingfilelistDamienTest);

    vector<Face_Bounding> imagesTest;
    //Steve
    Face_Bounding faceTest = {"steve","0",Point2d(122,106),Point2d(511,716),picturesTest[0]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve","0",Point2d(75,143),Point2d(462,661),picturesTest[1]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve","0",Point2d(138,132),Point2d(452,592),picturesTest[2]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve","0",Point2d(128,144),Point2d(436,604),picturesTest[3]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve","-45",Point2d(6,68),Point2d(159,78),picturesTest[4]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve","-45",Point2d(159,78),Point2d(569,591),picturesTest[5]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve","-30",Point2d(71,81),Point2d(530,650),picturesTest[6]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve","-30",Point2d(165,85),Point2d(518,580),picturesTest[7]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve","0",Point2d(111,74),Point2d(455,595),picturesTest[8]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve","0",Point2d(120,46),Point2d(443,515),picturesTest[9]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve","30",Point2d(100,39),Point2d(507,619),picturesTest[10]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve","30",Point2d(60,37),Point2d(426,527),picturesTest[11]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve","45",Point2d(111,56),Point2d(538,611),picturesTest[12]};
    imagesTest.push_back(faceTest);
    faceTest = {"steve","45",Point2d(3,47),Point2d(406,548),picturesTest[13]};
    imagesTest.push_back(faceTest);

    //Dan pictures

    faceTest = {"dan","0",Point2d(86,135),Point2d(492,622),picturesTest[14]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan","0",Point2d(118,106),Point2d(520,586),picturesTest[15]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan","0",Point2d(115,144),Point2d(434,566),picturesTest[16]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan","0",Point2d(108,158),Point2d(427,587),picturesTest[17]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan","0",Point2d(95,100),Point2d(509,671),picturesTest[18]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan","0",Point2d(102,142),Point2d(490,660),picturesTest[19]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan","30",Point2d(8,158),Point2d(477,779),picturesTest[20]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan","30",Point2d(30,120),Point2d(540,739),picturesTest[21]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan","45",Point2d(17,161),Point2d(512,757),picturesTest[22]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan","45",Point2d(44,98),Point2d(580,741),picturesTest[23]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan","-30",Point2d(57,135),Point2d(520,660),picturesTest[24]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan","-30",Point2d(105,156),Point2d(505,625),picturesTest[25]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan","-45",Point2d(56,90),Point2d(555,637),picturesTest[26]};
    imagesTest.push_back(faceTest);
    faceTest = {"dan","-45",Point2d(156,124),Point2d(576,622),picturesTest[27]};
    imagesTest.push_back(faceTest);

    //Thuy-Anh
    faceTest = {"thuy-anh", "0", Point2d(119,93),Point2d(461,615),picturesTest[28]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", "0", Point2d(149,191),Point2d(480,631),picturesTest[29]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", "0", Point2d(117,194),Point2d(480,565),picturesTest[30]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", "0", Point2d(115,190),Point2d(455,643),picturesTest[31]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", "0", Point2d(120,100),Point2d(475,550),picturesTest[32]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", "0", Point2d(142,116),Point2d(454,574),picturesTest[33]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", "-45", Point2d(119,132),Point2d(489,577),picturesTest[34]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", "-45", Point2d(144,148),Point2d(477,554),picturesTest[35]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", "-30", Point2d(96,138),Point2d(457,584),picturesTest[36]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", "-30", Point2d(182,124),Point2d(500,527),picturesTest[37]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", "30", Point2d(107,114),Point2d(413,548),picturesTest[38]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", "30", Point2d(89,139),Point2d(406,548),picturesTest[39]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", "45", Point2d(111,120),Point2d(480,600),picturesTest[40]};
    imagesTest.push_back(faceTest);
    faceTest = {"thuy-anh", "45", Point2d(66,118),Point2d(437,556),picturesTest[41]};
    imagesTest.push_back(faceTest);

    //Damien
     faceTest = {"damien","0",Point2d(122,138),Point2d(451,628),picturesTest[42]};
     imagesTest.push_back(faceTest);
     faceTest = {"damien","0",Point2d(154,136),Point2d(428,580),picturesTest[43]};
     imagesTest.push_back(faceTest);
     faceTest = {"damien","-30",Point2d(68,145),Point2d(481,686),picturesTest[44]};
     imagesTest.push_back(faceTest);
     faceTest = {"damien","-30", Point2d(142,134),Point2d(491,586),picturesTest[45]};
     imagesTest.push_back(faceTest);
     faceTest = {"damien", "-45",Point2d(305,106),Point2d(531,703),picturesTest[46]};
     imagesTest.push_back(faceTest);
     faceTest = {"damien","-45",Point2d(95,112),Point2d(554,652),picturesTest[47]};
     imagesTest.push_back(faceTest);
     faceTest = {"damien","30",Point2d(80,131),Point2d(505,721),picturesTest[48]};
     imagesTest.push_back(faceTest);
     faceTest = {"damien","30",Point2d(52,74),Point2d(425,638),picturesTest[49]};
     imagesTest.push_back(faceTest);
     faceTest = {"damien", "45",Point2d(72,133),Point2d(474,681),picturesTest[50]};
     imagesTest.push_back(faceTest);
     faceTest = {"damien", "45",Point2d(17,109),Point2d(409,611),picturesTest[51]};
     imagesTest.push_back(faceTest);
     faceTest = {"damien","0",Point2d(158,142),Point2d(449,565),picturesTest[52]};
     imagesTest.push_back(faceTest);
     faceTest = {"damien","0",Point2d(159,123),Point2d(472,606),picturesTest[53]};
     imagesTest.push_back(faceTest);
     faceTest = {"damien", "0",Point2d(125,125),Point2d(475,570),picturesTest[54]};
     imagesTest.push_back(faceTest);
     faceTest = {"damien", "0",Point2d(133,153),Point2d(449,608),picturesTest[55]};
     imagesTest.push_back(faceTest);

    cout << "entering lbp main" << endl;
    //lbp_main(images, imagesTest);


    //faceTagging();
    vector<vector<int>> histogramTestImages;
    generateHistograms(imagesTest,centers,histogramTestImages);

    //0 is -45, 1 is -30, 2 is 0,3 is 30, 4 is 45
    //rows represent training images and columns testing
    Mat confusionMatrixSteve = Mat::zeros(5, 5, CV_64F);
    Mat confusionMatrixDamien = Mat::zeros(5, 5, CV_64F);
    Mat confusionMatrixTA = Mat::zeros(5, 5, CV_64F);
    Mat confusionMatrixDan = Mat::zeros(5, 5, CV_64F);
    double newNorm;
    double norm = 10000;
    int x;
    int i;
    int index = 0;
    int good = 0;
    for(x = 0; x < histogramTestImages.size(); x++){
        for (i = 0; i < histogramsForFaces.size(); i++){
            //computes the euclidean distance between the two descriptors
            vector<int> test = histogramTestImages[x];
            vector<int> train = histogramsForFaces[i];
            newNorm = cv::norm(test, train, NORM_L2);
            //Store the best match with that descriptor
            if (newNorm < norm){
               norm = newNorm;
               index = i;
            }
        }
        //Pushes back the center to which the descriptor is matched. The centers
        //are in the same order as the descriptors
        string nameTest = imagesTest[x].name;
        string poseTest = imagesTest[x].pose;
        string nameResult = images[index].name;
        string poseResult = images[index].pose;
        Mat matrixToPass;
        if (nameTest.compare("steve") == 0){
            matrixToPass = confusionMatrixSteve;
        } else if (nameTest.compare("damien") == 0){
            matrixToPass = confusionMatrixDamien;
        } else if (nameTest.compare("thuy-anh") == 0){
            matrixToPass = confusionMatrixTA;
        } else if (nameTest.compare("dan")== 0){
            matrixToPass = confusionMatrixDan;
        }
        insertInConfusionMatrix(poseTest,poseResult, matrixToPass);
        if (nameTest.compare(nameResult) == 0){
            good = good + 1;
        }
        norm = 10000;
        index = 0;
    }

    cout <<  "Good" << to_string(good);
normalizeMatrices(confusionMatrixDamien,confusionMatrixDan, confusionMatrixSteve, confusionMatrixTA);

    return 0;
}

void normalizeMatrices(Mat confusionMatrixDamien, Mat confusionMatrixDan, Mat confusionMatrixSteve, Mat confusionMatrixTA){
    cout << "Damien\n";
    int g, h;
Mat confusionMatrixDamienNormalized = Mat::zeros(5, 5, CV_64F);
Mat confusionMatrixSteveNormalized = Mat::zeros(5, 5, CV_64F);
Mat confusionMatrixDanNormalized = Mat::zeros(5, 5, CV_64F);
Mat confusionMatrixTANormalized = Mat::zeros(5, 5, CV_64F);
    for (g = 0; g < 5; g++){
        double sum = 0;
        for (h = 0; h < 5; h++){
            sum = sum + confusionMatrixDamien.at<int>(g,h);
        }
        for (h = 0; h < 5; h++){
            double value = confusionMatrixDamien.at<int>(g,h);
            double goodvalue = value/sum;
            confusionMatrixDamienNormalized.at<double>(g,h) = goodvalue;
        }
    }

   for (g = 0; g < 5; g++){
        for (h = 0; h < 5; h++){
            printf( " %.3f  ", confusionMatrixDamienNormalized.at<double>(g,h));
        }
        cout << "\n";
    }
    cout << "Steve" << "\n";
    for (g = 0; g < 5; g++){
        double sum = 0;
        for (h = 0; h < 5; h++){
            sum = sum + confusionMatrixSteve.at<int>(g,h);
        }
        for (h = 0; h < 5; h++){
            double value = confusionMatrixSteve.at<int>(g,h);
            double goodvalue = value/sum;
            confusionMatrixSteveNormalized.at<double>(g,h) = goodvalue;
        }
    }
   for (g = 0; g < 5; g++){
        for (h = 0; h < 5; h++){
            printf( " %.3f  ", confusionMatrixSteveNormalized.at<double>(g,h));
        }
        cout << "\n";
    }
cout << "Dan" << "\n";
for (g = 0; g < 5; g++){
    double sum = 0;
    for (h = 0; h < 5; h++){
        sum = sum + confusionMatrixDan.at<int>(g,h);
    }
    for (h = 0; h < 5; h++){
        double value = confusionMatrixDan.at<int>(g,h);
        double goodvalue = value/sum;
        confusionMatrixDanNormalized.at<double>(g,h) = goodvalue;
    }
}
for (g = 0; g < 5; g++){
    for (h = 0; h < 5; h++){
        printf( " %.3f  ", confusionMatrixDanNormalized.at<double>(g,h));
    }
    cout << "\n";
}

cout << "TA" << "\n";
for (g = 0; g < 5; g++){
    double sum = 0;
    for (h = 0; h < 5; h++){
        sum = sum + confusionMatrixTA.at<int>(g,h);
    }
    for (h = 0; h < 5; h++){
        double value = confusionMatrixTA.at<int>(g,h);
        double goodvalue = value/sum;
        confusionMatrixTANormalized.at<double>(g,h) = goodvalue;
    }
}
for (g = 0; g < 5; g++){
    for (h = 0; h < 5; h++){
        printf( " %.3f  ", confusionMatrixTANormalized.at<double>(g,h));
    }
    cout << "\n";
}
}

void faceTagging(){
    const int NUM_IMAGES_PART1 = 5;
    const string IMG_NAMES_PART1[] = {"/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/IMG_5373.JPG", "/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/IMG_5374.JPG","/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/IMG_5375.JPG","/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/IMG_5376.JPG","/home/thuy-anh/CVisionProject2015/FaceRecognitionProject/IMG_5377.JPG"};
    // Load Part1 images
    vector<Mat> Images_part1;
    for(int i = 0; i < NUM_IMAGES_PART1; i++)
    {
        cout << IMG_NAMES_PART1[i];
    Images_part1.push_back( imread( IMG_NAMES_PART1[i] ) );
    }
    //detectAndDisplay(Images_part1[0]);
    //imshow("yo",Images_part1[0]);
    //waitKey(0);

}

void insertInConfusionMatrix(string poseTest, string poseResult, Mat &confusionMatrix){
    int i, j = 0;
    if (poseTest.compare("-45") == 0){
        j = 0;
    } else if (poseTest.compare("-30") == 0){
        j = 1;
    } else if (poseTest.compare("0") == 0){
        j = 2;
    } else if (poseTest.compare("30") == 0){
        j = 3;
    } else if (poseTest.compare("45") == 0){
        j =4;
    }
    if (poseResult.compare("-45") == 0){
        i = 0;
    } else if (poseResult.compare("-30") == 0){
        i = 1;
    } else if (poseResult.compare("0") == 0){
        i = 2;
    } else if (poseResult.compare("30") == 0){
        i = 3;
    } else if (poseResult.compare("45") == 0){
        i =4;
    }
    /*
    int valueToEnter = 0;
    if (i == j){
        valueToEnter = 1;
    } else {
        valueToEnter = 0;
    }*/
    confusionMatrix.at<int>(i,j) = confusionMatrix.at<int>(i,j) + 1;
}

void generateHistograms(vector<Face_Bounding> &faces, Mat &centers, vector<vector<int>> &histogramTestImages){

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
    int histogram[10];
    int t;
    for (t=0; t < 20; t++){
        histogram[t]=0;
    }
    int center;
    vector< vector<double> > centerOfDescriptors;
    int x;
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
    vector<int> histogramVector;
    for (p = 0; p < 20; p++){
    cout << histogram[p] << "\n";
    histogramVector.push_back(histogram[p]);
    }
    //cout << "hello";
    //Mat histogramToMat = Mat(1,20,CV_64F,histogram);
    histogramTestImages.push_back(histogramVector);
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

void Part1 (vector<Face_Bounding> &faces, vector<vector<int>> &histogramsForFaces, Mat &centers) {
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

    Mat extractedDescriptors1;
    //stores the SIFT descriptors for the first and second image
    FeatureDescriptor->compute(image1,keyPointsInBox,extractedDescriptors1);
    descriptors.push_back(extractedDescriptors1);
    descriptorsForEach.push_back(extractedDescriptors1);
    }
    Mat labels;

    kmeans(descriptors, 20, labels,
                TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
                   3, KMEANS_PP_CENTERS, centers);

    int j;
    for (j = 0; j < faces.size(); j++){
    //For each picture, we take its descriptors and put it in the bin with
    //the closest center
    double newNorm;
    double norm = 10000;
    int histogram[5];
    int t;
    for (t=0; t < 20; t++){
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
    vector<int> histogramVector;
    for (p = 0; p < 20; p++){
    cout << histogram[p] << "\n";
    histogramVector.push_back(histogram[p]);

    }
    //cout << "hello";
    //Mat histogramToMat = cv::Mat(1,20,CV_8UC3,histogram);
    //cout << histogramToMat.at<int>(5);
    //cout << "E = " << endl << " " << histogramToMat << endl << endl;
    //Mat E = Mat::eye(4, 4, CV_64F);
    //cout << "E = " << endl << " " << E << endl << endl;
    //histogramsForFaces.push_back(histogramToMat);
    histogramsForFaces.push_back(histogramVector);
    }
}

void lbp_recognition_results(vector<Face_Bounding> test_faces, vector<Face_Bounding> faces, Mat centres, vector< vector<int> > faces_as_codewords)
{
    //for each face
    Mat features;
    int errors = 0;
    for(int i=0; i<test_faces.size(); i++)
    {
        normalize(test_faces[i].image, test_faces[i].image, 0, 255, CV_MINMAX, CV_32F);
        Rect bounding_box = Rect(test_faces[i].top_left, test_faces[i].bottom_right);
        features = lbp_extract(Mat(test_faces[i].image, bounding_box), 10, 10);

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

        //codewords contains the test image's representation as a collection of code words
        //find nearest vector in faces_as_codewords, which is the representation of the training images
        int matched_face = nearest_face(codewords, faces_as_codewords);
        cout << test_faces[i].name << " " << faces[matched_face].name << " " << matched_face << endl;
        if(test_faces[i].name.compare(faces[matched_face].name) != 0)
        {
            errors += 1;
        }
    }
    cout << "errors: " << errors << endl;
}

int nearest_face(vector<int> test_hist, vector< vector<int> > trained_hists)
{
    int min_dist = norm(test_hist, trained_hists[0]);
    int nearest = 0;
    int test_dist;
    for(int i=1; i<trained_hists.size(); i++)
    {
        test_dist = norm(test_hist, trained_hists[i]);
        if(test_dist < min_dist)
        {
            min_dist = test_dist;
            nearest = i;
        }
    }
    return nearest;
}

vector< vector<int> > lbp_main(vector<Face_Bounding> faces, vector<Face_Bounding> test_faces)
{
    Mat descriptors;
    for(int i=0; i<faces.size(); i++)
    {
        //extract lbp features within the bounding box only and add them to the descriptors matrix
        normalize(faces[i].image, faces[i].image, 0, 255, CV_MINMAX, CV_32F);
        Rect bounding_box = Rect(faces[i].top_left, faces[i].bottom_right);
        Mat face = Mat(faces[i].image, bounding_box);
        descriptors.push_back(lbp_extract(Mat(faces[i].image, bounding_box), 10, 10));
        // vconcat(descriptors, lbp_extract(Mat(faces[i].image, bounding_box), 10, 10), descriptors);
    }

    //cluster that ish
    Mat labels, centres;
    descriptors.convertTo(descriptors, CV_32F);
    kmeans(descriptors, 50, labels,
                TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
                   3, KMEANS_PP_CENTERS, centres);

    vector< vector<int> > faces_as_codewords = lbp_cluster(descriptors, faces, centres);

    lbp_recognition_results(test_faces, faces, centres, faces_as_codewords);

    Mat group_shot = imread("IMG_5374.JPG", CV_LOAD_IMAGE_UNCHANGED);
    vector< vector<Point> > tagged_faces = face_detect_main(group_shot);
    face_tagging_results(group_shot, tagged_faces, faces, centres, faces_as_codewords);

    return faces_as_codewords;
}

void face_tagging_results(Mat group, vector< vector<Point> > tagged_faces, vector<Face_Bounding> faces, Mat centres, vector< vector<int> > faces_as_codewords)
{
    Mat tagged_face;
    Mat features;
    Mat output;
    group.copyTo(output);
    //for each face
    for(int i=0; i<tagged_faces.size(); i++)
    {
        //extract the actual face first
        tagged_face = Mat(group, Rect(tagged_faces[i][0], tagged_faces[i][1]));

        //normalize it
        normalize(tagged_face, tagged_face, 0, 255, CV_MINMAX, CV_32F);

        //extract them features
        features = lbp_extract(tagged_face, 10, 10);

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

        //id and then name of matched face 
        int matched_face = nearest_face(codewords, faces_as_codewords);
        cout << "match: " << matched_face << endl;
        string name = faces[matched_face].name;

        //write the name on the picture
        putText(output, name, tagged_faces[i][0], FONT_HERSHEY_PLAIN, 2, Scalar(255,0,255));
    }
    // resize(output, output, Size(800, 600));
    imshow("tagged", output);
    waitKey(0);
}


//returns a vector containing histograms of code words
//this is each image's representation as a histogram of code words
vector< vector<int> > lbp_cluster(Mat lbp_features, vector<Face_Bounding> faces, Mat centres)
{
    // //clustering descriptors
    // Mat labels;
    // lbp_features.convertTo(lbp_features, CV_32F);
    // kmeans(lbp_features, 50, labels,
    //             TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
    //                3, KMEANS_PP_CENTERS, centres);

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

    return faces_codewords;
}

//get the nearest cluster centre, ie. to which cluster does this feature belong
int nearest_centre(Mat row, Mat centres)
{
    int nearest = 0;
    row.convertTo(row, CV_32F);
    centres.convertTo(centres, CV_32F);
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
    int row_height = face.rows / (double)W;
    int col_width = face.cols / (double)H;
    vector<int> histogram;
    Mat lbp_feature_row;
    Mat lbp_features = Mat(1, 256, CV_32F, 1);
    Point2d left;
    Point2d right;

    //loop over image and perform lbp junk on each window of face
    for(int i=0; i<W; i++)
    {
        for(int j=0; j<H; j++)
        {
            left = Point2d(j*col_width, i*row_height);
            right= Point2d(j*col_width+col_width, i*row_height+row_height);
            window = Mat(face, Rect(left, right));
            histogram = lbp_histogram(window);

            lbp_feature_row = Mat(histogram, true);
            lbp_feature_row = lbp_feature_row.t();
            lbp_feature_row.convertTo(lbp_feature_row, CV_32F);
            // for(int k=0; k<256; k++)
            //     lbp_feature_row.at<int>(0,k) = histogram[k];
            // vconcat(lbp_features, lbp_feature_row, lbp_features);
            lbp_features.push_back(lbp_feature_row);
        }
    }
    lbp_features = Mat(lbp_features, Rect(0,1,lbp_features.cols, lbp_features.rows-1));

    return lbp_features;
}

//returns the histogram of lbp descriptors, given one of the W x H windows of the image
vector<int> lbp_histogram(Mat img_window)
{
    //initialize histogram
    vector<int> hist;
    for(int i=0; i<256; i++)
        hist.push_back(0);

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
    // cvtColor(img, img, CV_RGB2GRAY);
    int center = img.at<int>(i,j);
    int out = 0;
    out += (img.at<int>(i-1, j) < center);
    out += (img.at<int>(i-1, j+1) < center)*2;
    out += (img.at<int>(i, j+1) < center)*4;
    out += (img.at<int>(i+1, j+1) < center)*8;
    out += (img.at<int>(i+1, j) < center)*16;
    out += (img.at<int>(i+1, j-1) < center)*32;
    out += (img.at<int>(i, j-1) < center)*64;
    out += (img.at<int>(i-1, j-1) < center)*128;
    return out;
}

vector< vector<Point> > face_detect_main(Mat frame)
{
    //to find where your module is, type "locate haarcascade_frontalface_alt.xml in your cmd line "
    string face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
    CascadeClassifier face_cascade;  
    vector < vector<Point> > faceDetectionVector;

    // Load the cascade
    if (!face_cascade.load(face_cascade_name))
    {
        printf("Couldn't load face cascade module\n");
        exit(-1);
    };

    // Read the image file
    // Mat frame = imread("group.JPG");
    //Mat frame = imread("lenna.png");

    if (!frame.empty())
    {
        faceDetectionVector = detectAndDisplay(frame, face_cascade); //the magic happens here : detects faces from a pic
        waitKey(0);
    }
    else
        printf("Are you sure you have the correct filename?\n");

    // cout << faceDetectionVector.size() << endl;
    return faceDetectionVector;
}

// detecs faces and displays them in the picture
vector < vector<Point> > detectAndDisplay(Mat frame, CascadeClassifier face_cascade)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    Mat crop;
    Mat res;
    Mat gray;
    vector<Point> faceCoords;
    vector < vector<Point> > faceVector;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    size_t ic = 0; // ic is index of current element

    for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)
    {
        Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
        faceCoords.push_back(pt1);
        Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
        faceCoords.push_back(pt2);
        rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);

        faceVector.push_back(faceCoords);
        faceCoords.clear();
    }

    // imshow("original", frame);

    return faceVector;
    //return coords;

}
