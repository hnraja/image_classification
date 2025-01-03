# object_detection_classification

Limitations
* can only identify images of the following
    * planes
    * cars
    * birds
    * cats
    * deer
    * dogs
    * frogs
    * horses
    * ships
    * trucks
* since model is trained on built-in keras datasets containing only these objects 


File Tree
* model.py 
    * code used to create model.keras
    * fits multiple models and selects best model based on accuracy
    * long run time (9+ hours), since multiple models tested
    * save model metrics to performance.csv
* main.py 
    * uses model.keras to identify the images in ./images
* images/
    * image source: https://pixabay.com/
    * contains images for testing
* performance.csv
    * contains information about models fit and their performance
* performance_eval.py
    * identify best models
    * summarized in performance_eval.txt
    * generates line charts for comparing models 
* performance_eval.txt
    * observations about models
* dummy.py
    * testing small code snippets
    * to be deleted at completion


Learning Objectives
* Keras datasets
* Keras models
* Keras layers

Room for improvements
* magic numbers in performance_eval.py