// CSIRO has filed various patents which cover the Software. 

// CSIRO grants to you a license to any patents granted for inventions
// implemented by the Software for academic, research and non-commercial
// use only.

// CSIRO hereby reserves all rights to its inventions implemented by the
// Software and any patents subsequently granted for those inventions
// that are not expressly granted to you.  Should you wish to license the
// patents relating to the Software for commercial use please contact
// CSIRO IP & Licensing, Gautam Tendulkar (gautam.tendulkar@csiro.au) or
// Nick Marsh (nick.marsh@csiro.au)

// This software is provided under the CSIRO OPEN SOURCE LICENSE
// (GPL2) which can be found in the LICENSE file located in the top
// most directory of the source code.

// Copyright CSIRO 2013

#include <iostream>
#include <fstream>
#include <sstream>

#include "utils/helpers.hpp"
#include "utils/command-line-arguments.hpp"
#include "utils/points.hpp"
#include "tracker/FaceTracker.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace FACETRACKER;
using namespace std;
using namespace cv;

const char* ann_xml = "ann.xml";
const char* csv_file = "label.csv";

/* 23 distances between facial points */
#define INPUT_SIZE 23

/* 7 different expressions */
#define OUTPUT_SIZE 7

float get_distance(Point_<double> a, Point_<double> b);
void get_euler_distance_sets(vector<Point_<double> >& points, vector<double>& distances);
int get_facial_points(Mat& face, vector<Point_<double> >& points);

void read_training_data_from_csv(String csvfile, Mat& data, Mat& label, char separator=';')
{
    std::ifstream file(csvfile.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }

    string line, path, classlabel;
    int i = 0;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            Mat face = imread(path, 1);
            vector<Point_<double> > points;
            vector<double> distances;
            if(!get_facial_points(face, points)) {
                continue;
            }

            get_euler_distance_sets(points, distances);
            int j = 0;
            while(!distances.empty()) {
                data.at<double>(i,j) = distances.back();
                distances.pop_back();
                j++;
            }

            for (j = 0; j < 7; j++) {
                if (j == atoi(classlabel.c_str()))
                    label.at<double>(i, j) = 1;
                else
                    label.at<double>(i, j) = 0;
            }
        }
        cout<<"reading " << i<<"\n";

        i++;
    }
    data.resize(i);
    label.resize(i);
}

void train_ann(String csv, String ann_file)
{
    Mat layers(3,1,CV_32S);
    layers.at<int>(0,0) = INPUT_SIZE; //input layer
    layers.at<int>(1,0) = 40; //hidden layer
    layers.at<int>(2,0) = OUTPUT_SIZE; //output layer

    CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM,0.6,1);
    CvANN_MLP_TrainParams params(cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001),
                                 CvANN_MLP_TrainParams::BACKPROP,
                                 0.1, 0.1);

    Mat data(400, 23, CV_64FC1), label(400, 7, CV_64FC1);
    read_training_data_from_csv(csv, data, label);
    nnetwork.train(data, label,cv::Mat(),cv::Mat(),params);

    FileStorage fs(ann_file, cv::FileStorage::WRITE); // or xml
    nnetwork.write(*fs, "facial_ann"); // don't think too much about the deref, it casts to a FileNode
}

float get_distance(Point_<double> a, Point_<double> b)
{
    return sqrt(((a.x - b.x) * (a.x - b.x)) + (a.y - b.y) * (a.y - b.y));
}

void get_euler_distance_sets(vector<Point_<double> >& points, vector<double>& distances)
{
    distances.push_back(get_distance(points[17], points[41]));
    distances.push_back(get_distance(points[19], points[41]));
    distances.push_back(get_distance(points[21], points[41]));

    distances.push_back(get_distance(points[22], points[42]));
    distances.push_back(get_distance(points[24], points[42]));
    distances.push_back(get_distance(points[26], points[42]));

    distances.push_back(get_distance(points[36], points[41]));
    distances.push_back(get_distance(points[36], points[31]));
    distances.push_back(get_distance(points[41], points[31]));

    distances.push_back(get_distance(points[42], points[45]));
    distances.push_back(get_distance(points[42], points[25]));
    distances.push_back(get_distance(points[45], points[25]));

    distances.push_back(get_distance(points[31], points[51]));
    distances.push_back(get_distance(points[25], points[51]));

    distances.push_back(get_distance(points[41], points[51]));
    distances.push_back(get_distance(points[51], points[54]));

    distances.push_back(get_distance(points[41], points[58]));
    distances.push_back(get_distance(points[58], points[54]));

    distances.push_back(get_distance(points[31], points[41]));
    distances.push_back(get_distance(points[58], points[31]));

    distances.push_back(get_distance(points[25], points[58]));
    distances.push_back(get_distance(points[25], points[54]));

    distances.push_back(get_distance(points[58], points[51]));
}

int get_facial_points(Mat& face, vector<Point_<double> >& points)
{
    FaceTracker * tracker = LoadFaceTracker(DefaultFaceTrackerModelPathname().c_str());
    FaceTrackerParams *tracker_params  = LoadFaceTrackerParams(DefaultFaceTrackerParamsPathname().c_str());

    Mat frame_gray;
    cvtColor(face, frame_gray, CV_RGB2GRAY );

    int result = tracker->NewFrame(frame_gray, tracker_params);

    vector<Point_<double> > shape;
    Pose pose;

    if (result >= 1) {
        points = tracker->getShape();
        pose = tracker->getPose();
    } else {
        return 0;
    }

    delete tracker;
    delete tracker_params; 

    return 1;
}

int classify_emotion(Mat face, const char* ann_file)
{
    int ret = 0;
    Mat output(1, OUTPUT_SIZE, CV_64FC1);
    Mat data(1, INPUT_SIZE, CV_64FC1);
    CvANN_MLP nnetwork;
    nnetwork.load(ann_file, "facial_ann");

    vector<Point_<double> > points;
    vector<double> distances;
    if(!get_facial_points(face, points)) {
        return -1;
    }

    imshow("face", face);
    get_euler_distance_sets(points, distances);
    int j = 0;
    while(!distances.empty()) {
        data.at<double>(0,j) = distances.back();
        distances.pop_back();
        j++;
    }

    nnetwork.predict(data, output);

    double b = 0;
    int k = 0;
    for (j = 0; j < 7; j++) {
        cout<<output.at<double>(0, j)<<" ";
        if (b < output.at<double>(0, j)) {
            b = output.at<double>(0, j);
            k = j;
        }
    }
    cout<<"\n";

    return k;
}

String get_emotion(int e)
{
    switch (e) {
        case 0:
            return "Neutral";
        case 1:
            return "Anger";
        case 2:
            return "Contempt";
        case 3:
            return "Disgust";
        case 4:
            return "Fear";
        case 5:
            return "Happy";
        case 6:
            return "Sad";
        case 7:
            return "Surprise";
        default:
            return "Unknown";
    }

    return "Unknown";
}

int main(int argc, char* argv[])
{
    Mat frame; 
    if (argv[1][0] == 't') {
        train_ann(csv_file, ann_xml);
        return 0;
    } else if (argv[1][0] == 'f') {
        int ret;
        ret = classify_emotion(imread(argv[2], 1), ann_xml);
        cout << get_emotion(ret) << "\n";
    } else if (argv[1][0] == 'c') {
        CvCapture* capture;

        capture = cvCaptureFromCAM( 0 );
        if(capture) {
            while(true) {
                frame = cvQueryFrame(capture);
                if(!frame.empty()) {
                    int ret;
                    ret = classify_emotion(frame, "./ann.xml");
                    cout << get_emotion(ret) << "\n";
                } else {
                    printf(" --(!) No captured frame -- Break!");
                    break;
                }

                int c = waitKey(10);
                if((char)c == 'c') {
                    break;
                }
            }
        }
    }

    for(;;) {
        int c = waitKey(10);
        if((char)c == 'c') {
            break;
        }
    }
    return 0;
}
