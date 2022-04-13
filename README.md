# Image-segmentation-on-GPU
The project aims at studying the execution time and performance of K-means image
segmentation algorithm on GPU as well as CPU. Image segmentation is a technique to reduce
the complexity of the image by dividing the image into various subgroups. K-means algorithm
does a decent job in segmenting the image based on the locality of the pixels relative to the
randomly assigned centroids. The centroids shift as the pixels are assigned to their respective
nearest centroids. As the algorithm is highly parallelizable, it is implemented on the GPU. The
results are found to be as expected- the sample images performed significantly better on GPU.
