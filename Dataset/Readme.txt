We have three dataset and the accroding codes.

Folder Synthetic_Data
We write code to generate the Twinmoon data.
Twin_moon.ipynb: Generate Twinmoon data with file name Twin_moon.npy
Twin_moon_Sp.ipynb: Generate Twinmoon data with different pattern for special case with file name Twin_moon.npy

Folder Office
The provider provides the data in .mat format. We need to translate it into pyhton format.
To run the code, please place the the .mat data right in this folder.
mat2npy.ipnyb :  This code can translate the .mat into .npy which can be used for python. 
                 Here also splits the data into training set, validation set and testing set.

Folder Amazon_Reviews
Here we translate the origin data into bag-of-words unigram/bigram features and processed by tf-idf. We use scikit-learn tool to run tf-idf
The origin data is one line per document, with each line in the format:
feature:<count> .... feature:<count> #label#:<label>       #features refer to unigram/bigram
To run the code, please place the four folder(books, dvd, electronics, kitchen, each contatin negative.review, positive.review and unlabeled.review) in this folder.

Preprocess_file2corpus.ipynb : 
This code tanslate the origin features into the format for scikit-learn and pick up the label value.
It will package the result of negative.review, positive.review and unlabeled.review into one pickle.

Preprocess_corpus2tfidf_seperate.ipynb : 
This code use the previous result pickle and run tf-idf process.
It will pick reviews from any two domains and run tf-idf process.                                         

Preprocess_corpus2tfidf_union.ipynb : 
This code use the previous result pickle and run tf-idf process.
It will use reviews from all domains and run tf-idf process.

The difference of "seperate" and "union" is the scope of the useing corpus.