### Implementing Matrix Factorization

Use the [instructor's notebook](https://github.com/nipunbatra/ml-teaching/blob/master/notebooks/movie-recommendation-knn-mf.ipynb) on matrix factorisation, and solve the following questions.

**a) Image Reconstruction-** Here, ground truth pixel values are missing for particular regions within the image- you don't have access to them.

-  Use an image and reconstruct the image in the following two cases, where your region is-
    1. a rectangular block of 30X30 is assumed missing from the image. 
    2. a random subset of 900 (30X30) pixels is missing from the image. 

    Choose rank `r` yourself. Perform Gradient Descent till convergence, plot the selected regions, original and reconstructed images, Compute the following metrics:
    * RMSE on predicted v/s ground truth high resolution image
    * Peak SNR

    
-  Write a function using this [reference](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html) and use alternating least squares instead of gradient descent to repeat Part 1, 2 of Image reconstruction problem using your written function. 

**b) Data Compression-** Here, ground truth pixel values are not missing- you have access to them. You want to explore the use of matrix factorisation in order to store them more efficiently.
- Consider an image patch of size (NxN) where N=50. We are trying to compress this patch (matrix) into two matrices, by using low-rank matrix factorization. Consider the following three cases-
    1. a patch with mainly a single color.
    2. a patch with 2-3 different colors.
    3. a patch with at least 5 different colors.

    Vary the low-rank value as ```r = [5, 10, 25, 50]```  for each of the cases. Use Gradient Descent and plot the reconstructed patches over the original image (retaining all pixel values outside the patch, and using your learnt compressed matrix in place of the patch) to demonstrate difference in reconstruction quality. Write your observations. 

