# Face-Generator
As AI continues to advance in the 21st century, with the advent of faster and more available CPU’s and Processing power, computer vision and image generation have been brought into the field of practicality. As we advance in these steps, multiple techniques are created and refined. One of the current main techniques involves using Generative Adversarial Networks that rely on a generator and a discriminator to perfect each other and produce more and more lifelike images. In this project, I took a dive into a [recently published paper regarding Pro-GANs](https://arxiv.org/abs/1710.10196). In this project I attempt to implement this version of a GAN. The second part consists of a face generator constructed relying on a pre-built model by NVIDIA with minor adjustments on input and output handling. 

## ProGAN

A ProGAN is similar to a GAN but in the sense the images are started at low res before being scaled up by a factor each time to achieve high res pictures. I instead went with a size cap of 256 to save on computer resources. GAN's are known as Generative Adversarial Networks and rely on a generator and discriminator for the construction of an image. The generator constructs an image based upon a random input, and will attempt to fool the discriminator which has to classify if the image is fake. As the two continue to fight one another, it leads to lifelike image construction. ProGAN's differ from normal GAN's by the basis of running starting at a low res image of 16 before upscaling to 32, 64, 128, 256 etc. The concept is explained in more detail [in the paper](https://arxiv.org/abs/1710.10196).

## Nvidia's Pre-Built solution
Face Generator Using Nvidia's NN's. While training an GAN on a personal machine would've been preferred, current resources did not allow for that. So instead, I used the already created Nvidia GAN and slightly modified its generation and strung togegther the produced images to create a small clip of a changing face. The final result can be seen [![Here]](https://www.youtube.com/watch?v=uh920Nd_kgk).


## Notes

This was a simple implementation exercise and all credits for the concepts of a ProGAN and the construction of the NVIDIA model belong to their respective owners.
