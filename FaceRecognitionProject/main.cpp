//Thuy-Anh Le - 260455284
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
using namespace cv;
using namespace std;
// Functions prototypes
void Part1(vector<Mat> &Images);
void Part2(vector<Mat> &Images);
Mat RANSACDLT(vector<Point2d> keypoints1, vector<Point2d> keypoints2);
int main()
{
// Initialize OpenCV nonfree module
initModule_nonfree();
// Set the dir/name of each image for Part1 here
const int NUM_IMAGES_PART1 = 1;
const string IMG_NAMES_PART1[] = {"/home/thuy-anh/FaceRecognitionProject/TA1.jpg"};
// Set the dir/name of each image for Part1 here
const int NUM_IMAGES_PART2 = 2;
const string IMG_NAMES_PART2[] = {"/home/thuy-anh/Assignment3CompVision/plan1.jpg", "/home/thuy-anh/Assignment3CompVision/plan2.jpg"};
// Load Part1 images
vector<Mat> Images_part1;
for(int i = 0; i < NUM_IMAGES_PART1; i++)
{
Images_part1.push_back( imread( IMG_NAMES_PART1[i] ) );
}
// Load Part2 images
vector<Mat> Images_part2;
for(int i = 0; i < NUM_IMAGES_PART2; i++)
{
Images_part2.push_back( imread( IMG_NAMES_PART2[i] ) );
}
// Call Part1 function
Part1(Images_part1);
// Call Part2 function
return 0;
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
void Part1 (vector<Mat> &Images) {
    Mat image1 = Images[0];
    Ptr<FeatureDetector> FeatureDetectorSIFT = FeatureDetector::create("SIFT");
    vector <KeyPoint> keyPoints1;

    //Stores the keypoints for the first and second image
    FeatureDetectorSIFT->detect(image1, keyPoints1);


    Ptr<DescriptorExtractor> FeatureDescriptor = DescriptorExtractor::create("SIFT");

    vector<vector<Point2d>> boundingBoxes;
    Point2d point1;
    Point2d point2;
    point1.x = 119;
    point1.y = 111;
    point2.x = 437;
    point2.y = 577;
    vector<Point2d> rectangle;
    rectangle.push_back(point1);
    rectangle.push_back(point2);
    boundingBoxes.push_back(rectangle);

    //Discard the keypoints that are outside of the bounding box
    int i;
    vector<int> keyPointsInBoxIndex;
    vector<KeyPoint> keyPointsInBox;
    cv::rectangle(image1,boundingBoxes[0][0],boundingBoxes[0][1],1,8,0);
    //point1.x =
    for (i = 0; i < keyPoints1.size(); i++){
        //if keypoint is inside then
        bool inBox = in_box(keyPoints1[i].pt,boundingBoxes[0][0],boundingBoxes[0][1]);
        if (inBox){
                keyPointsInBoxIndex.push_back(i);
                keyPointsInBox.push_back(keyPoints1[i]);
        }
    }

    Mat outputImageSIFT;
    //drawKeypoints(image1, keyPointsInBox, outputImageSIFT,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //imshow("Hello",outputImageSIFT);
    //waitKey(0);

    Mat extractedDescriptors1;
    vector<vector<double>> extractedDescriptorsInBox;
    //stores the SIFT descriptors for the first and second image
    FeatureDescriptor->compute(image1,keyPointsInBox,extractedDescriptors1);
    int y;
    for (y = 0; y < extractedDescriptors1.rows; y++){
       extractedDescriptorsInBox.push_back(extractedDescriptors1.row(y));
    }
    Mat labels, centers;
    kmeans(extractedDescriptors1, 50, labels,
                TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
                   3, KMEANS_PP_CENTERS, centers);
    cout << "hello";
    double newNorm;
    double norm = 1000;
    int histogram[extractedDescriptors1.rows];
    int t;
    for (t=0; t < extractedDescriptors1.rows; t++){
        histogram[t]=0;
    }
    vector<int> histo;

    int center;
    vector<vector<double>> centerOfDescriptors;
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
        //histogram[center] = histogram[center] + 1;
        vector<int>::iterator iterator = histo.begin();
        histo.insert(iterator +1, 2,2);
        norm = 1000;
        center = 0;
    }
   /* int p;
    for (p = 0; p < sizeof(histogram); p++){
    cout << histogram[p] + "\n";
    } */
    cout << "hello";

}


//determines if the first point is within the box defined by the second two points

