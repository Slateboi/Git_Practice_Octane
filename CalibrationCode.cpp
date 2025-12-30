// # g++ CameraCalibration.cpp $(pkg-config --cflags --libs opencv4) -o Calibration
// ./Calibration


#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> 
#include <opencv2/imgproc.hpp>   

int main( int argc ,char **argv) { 
    (void)argc;
    (void)argv;

    std::vector<cv::String> fileNames; // reads img path in filenames and storing in array vector
    cv::glob("imgs/*.jpeg", fileNames, false); // false means do not search subdirectories

    // Debug Output for accessing files/images correctly
    std::cout << "Found " << fileNames.size() << " images:" << std::endl;
    for(auto& f : fileNames) {
    // std::cout << "  " << f << std::endl;
    }
    if(fileNames.empty()) {
    std::cout << "NO IMAGES FOUND! Check folder name and .jpeg files." << std::endl;
    return -1;
    }

    // cv::Size is a simple OpenCV struct that stores a 2D size: width and height
    cv::Size patternSize(9,9); // (col,row) --> represent 24,18 inner corner points 

    /* Basics 
    std::vector<cv::Point2f> = all detected 2D corners in one image.
    std::vector<std::vector<cv::Point2f>> = for all images (array of arrays)
    */

    std::vector<std::vector<cv::Point2f>> q(fileNames.size()); // 2D point with float coordinates (x,y), typically pixel locations in the image plane 
    //So q[i] will hold all detected chessboard corners in image i (pixel coordinates
    std::vector<std::vector<cv::Point3f>> Q;

    //1. Generate checker board (world) coordinates Q. The board has 25x 18
    // fields with a size of 15x15 mm

    int checkerBoard[2] = {9,9};
    int fieldSize = 16;

    //Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;  // template
    for(int i = 0; i < checkerBoard[1]; i++) {
    for(int j = 0; j < checkerBoard[0]; j++) {
        objp.push_back(cv::Point3f(
            j * fieldSize,
            i * fieldSize,
            0
        ));
    }
}

    //Detect feature points
    std::vector<cv::Point2f> imgPoint;
    std::size_t i = 0;

    // auto converts the cv::String (fileNames) to std::string automatically
    // const& f is the reference for each string in the fileName (efficient)
    for(auto const &f : fileNames){ 

    std::cout<< std::string(f) << std::endl;

    //2. Read in the Image by cv::findChessboardCorners()
    cv::Mat img = cv::imread(f); // ? cv::imread(f) /// fileNames[i]
    cv::Mat gray;
    cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);
    bool patternFound = cv::findChessboardCorners(gray,patternSize,q[i],cv::CALIB_CB_ADAPTIVE_THRESH  | cv::CALIB_CB_NORMALIZE_IMAGE ); // detects the internal corner points of a chessboard calibration pattern in an image and returns their 2D pixel coordinates.​

    std::cout << "patternFound = " << patternFound
          << " | detected points = " << q[i].size() << std::endl;


    /* cv::findChessboardCorner() '''
   -Finds the Chessboard's Internal Corners as defined 
    in the patternSize and stores pixel coordinates in q[i]

   -cv::CALIB_CB_ADAPTIVE_THRES --> Applies Adaptive Threshold 
    befroe Searching the Internal Corners

   -cv::CALIB_CB_NORMALIZE_IMAGE --> Normalize the Image brightness
    and Contrast beffore detection 

   -cv::CALIB_CB_FAST_CHECK --> Performs a fast if the pattern is 
    a chessboard-type pattern, which is cheap and a fast check!    

    */


    //3. Use cv::cornerSubPix() to refine the found corners
    if(patternFound){
        std::cout<<"Chessboard detected!"<<std::endl;
        cv::cornerSubPix(gray,q[i],cv::Size(11,11),cv::Size(-1,-1),cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,30,0.1)); // 30,0.1 ?
        Q.push_back(objp); // ? why to copy objp in Q
        }
    
    /* cv::cornerSubPix()
   -An Algorithm which refines the values of q[i]
   
   -Gets the exact pixel coordinate of the Internal Corner 
    coordiantes as detected in the cv::findChessboardCorner()

   -Gets the exact pixel coordinates in precision of floating No.

   -Explain Parameters Passed
    */
    
    //Display

    /*cv::drawChessboardCorner()

   -Draws a Connect Colored grid of all the Internal Corners 
    points,if all the points mentioned in patternSize are found

   -If not then the detected Internal Corners points are drawn 
    in simple circle

   -If the bool(patternFound is false) no pattern is drawn
    
    */
    cv::drawChessboardCorners(img,patternSize,q[i],patternFound);
    cv::imshow("Chess Board Detection",img);
    cv::waitKey(0);

    i++;
    
    }
//  cv::Matx33f::eye() --> creates a 3x3 Identity Matrix
    cv::Matx33f K(cv::Matx33f::eye()); // instrinsic matrix 
//  cv::Vec<T, n> is a fixed‑length vector type in OpenCV with n elements of type T
    cv::Vec<float,5> k(0,0,0,0,0); // distortion coefficients

    std::vector<cv::Mat> rvecs ,tvecs;
    std::vector<double> stdIntrinsics ,stdExtrinsics, perViewErrors;

    int flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 + cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT;

    /* int flags
   -cv::CALIB_FIX_ASPECT_RATIO --> keeps the ratio of fx/fy 
    const (Aspect ratio),also implies that the pixels are square
    
   -cv::CALIB_FIX_K3 --> keeps K3 parameter constant that implies 
    it ignores higher order radial distortion 

   -cv::CALIB_ZERO_TANGENT_DIST --> keep the tangential distortion 
    parameter (p1,p2) as constant it implies that images is not skewed 
    (lens is not mounted tilted)

   -cv::CALIB_FIX_PRINCIPAL_POINT --> keeps the principal pt (Cx,Cy)
    at the centre of the image    

    */
    cv::Size frameSize(1440,1080); //framesize of the output image

    std::cout<<"Calibrating..."<<std::endl;

    //4. Call "float error = cv::calibrateCamera()" with the input coordinates
    if (Q.empty()) {
    std::cerr << "ERROR: No valid chessboard detections found!" << std::endl;
    return -1;
    }

    float error = cv::calibrateCamera(Q,q,frameSize,K,k,rvecs,tvecs,flags);

    /*cv::calibrateCamera()

   -It estimates : 1.Camera Instrinsics, 2.Lens Distortion 3.Pose (rvecs,tvecs) 

   -Q --> template
   -q --> Detected Interanl Corner Points
   -K --> Intrinsic Matrix (Intially passed as Identity Mat) 
   -k --> Distortion Coefficients (k1,k2,p1,p2,k3) 

    */

    std::cout<<"Reprojection error = "<< error <<"\nK =\n"
        << K <<"\nk=\n"
        << k << std::endl;


    //precompute lens correction interpolation

    cv::Mat mapX,mapY ;
    cv::initUndistortRectifyMap(K,k,cv::Matx33f::eye(),K,frameSize,CV_32FC1,mapX,mapY);

    /*cv::initUnidistortRectifyMap()
   -precomputes two maps (mapX, mapY) that tell, for every pixel 
    in the undistorted image, which pixel to sample from in the 
    original distorted image.  

   -MapX --> X coordinate in the distorted image (Datatype: CV_32FC1)
   -MapY --> Y coordinate in the distorted image (Datatype: CV_32FC1)

   -Both MapX,MapY are important pixel coordinates 
    to Recosntruct the Undistorted Image

   -CV_32FC1 --> 32 Bit floating point mumber with channel only 1
    that is MapX could only store single values (only X coordinates)

   -1st K --> passing the Camera Intrinsics
   -2nd K --> setting the Intrinsic Parameters for the Generated 
    Undistorted image to be same as Distorted Image i.e. K

   -cv::Matx33f::eye() --> Refers to Rectification Matrix,it states 
    how to Rotate the Camera View before making the Undistoreted Images 
    by passing the Identity Matrix == No Rotations/Tilts
        
    */

    //Show lens corrected img
    for(auto const& f : fileNames){
        std::cout<< std::string(f) <<std::endl;
        cv::Mat img = cv::imread(f,cv::IMREAD_COLOR);

        //5. Remap the image being using the precomputed interploation map
        cv::Mat imgUndistorted;
        cv::remap(img,imgUndistorted,mapX,mapY,cv::INTER_LINEAR);

        /*cv::remap()

       -Combines the image coordinates in Mapx ,MapY in an 
        Interpolation method using Bilinear Interpolation 

      -cv::INTER_LINEAR --> Bilinear Interpolation uses 4 
       neighbouring pixel coordinates to smooth out the Values
       
        */


        //Display
        cv::imshow("Undistored image",imgUndistorted);
        cv::waitKey(0);
        
        }
    return 0;
               
}
