
Image = imread('parrot.jpg');
imshow(Image)
title('Original Image')
tic
[Label,Centers] = imsegkmeans(Image,8);
B = labeloverlay(Image,Label);
toc
imshow(B)
title('Segmented Image')
