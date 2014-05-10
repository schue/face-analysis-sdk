This tool detects the facial emotion from either webcam or an image using a neural network.

A pre-trained ANN result stores in {SRC}/data_set/ann.xml, trained with Cohn-Kanade database(http://www.pitt.edu/~emotion/ck-spread.htm).

You may train your own ANN with a CSV file which each line of the trainning data is: "\<path-to-image\>; \<emotion label\>". The emotions are tagged in the following way: 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise.

The idea was mostly based on "Automatic Facial Expression Recognition System Based on Geometric and Appearance Features" written by Aliaa A. A. Youssif & Wesam A. A. Asker. Using the AAM in face-analysis-sdk to get 16 facial points from the face image, then calculate the euler distance of these points and use them as the input of the neural network.

3 points on each eyebrow, 2 points on each eye, 2 points on the nose holes, 4 points on the mouth.

    17   19   21              22   24   26
     +    +    +               +    +    +
                                        
      36      41              42      45       
       +       +               +       +
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                  31    25        
                   +     +             
                          
                     51     
                      +                
              41            54
               +             +
                     58
                      +                
