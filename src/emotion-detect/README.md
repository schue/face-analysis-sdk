### Introduction

This tool detects the facial emotion from either webcam or an image using a neural network.

### Usage

    Usage: <running mode> <arguments>

    Trainning mode:
    $emotion-detect t <path-to-ann-file> <path-to-trainning-label-csv-file>
    Image mode:
    $emotion-detect f <path-to-ann-file> <path-to-image>
    Webcam mode:
    $emotion-detect c <path-to-ann-file>

A pre-trained ANN result stores in {SRC}/data_set/ann.xml, trained with [Cohn-Kanade database](http://www.pitt.edu/~emotion/ck-spread.htm).

You may train your own ANN with a CSV file which each line of the trainning data is: "\<path-to-image\>; \<emotion label\>". The emotions are tagged in the following way: 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise.

### How to compile

It will be compiled along with the sdk.
    
    cd {SDK-BASE-DIR}
    cmake .
    make

The executable binary will be in {SDK-BASE-DIR}/bin.

### Implementation

The idea was mostly based on "Automatic Facial Expression Recognition System Based on Geometric and Appearance Features" written by Aliaa A. A. Youssif & Wesam A. A. Asker. Using the AAM in face-analysis-sdk to get 16 facial points from the face image, then calculate the euler distance of these points and use them as the input of the neural network.

3 points on each eyebrow, 4 points on each eye, 3 points on the nose, 4 points on the mouth.

    17   19   21              22   24   26
     +    +    +               +    +    +
          
          37                      43
           +                       +
      36      39              42      45       
       +       +               +       +
          41         27           47         
           +          +            +      
                                        
                                        
                                        
                                        
                                        
                  31    25        
                   +     +             
                          
                     51     
                      +                
              41            54
               +             +
                     58
                      +                

Connect the points pairs in the following way, and use these distances as the input of the neural network.

![A demo shows the distances used for neural network](https://raw.githubusercontent.com/supermartian/face-analysis-sdk/61ce2983f727cd13efd95e80b87f50c0bd70af61/src/emotion-detect/face_demo.png)
