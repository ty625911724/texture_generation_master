# texture_generation_master

Implementation of [INCORPORATING LONG-RANGE CONSISTENCY IN CNN-BASED TEXTURE GENERATION](https://arxiv.org/pdf/1606.01286.pdf) by Tensorflow.

## Requirements
 - [Tensorflow]
 - [Pillow]
 - [scipy]
 - [VGG 19 model](https://drive.google.com/file/d/0B8QJdgMvQDrVU2cyZjFKU1RrLUU/view?usp=sharing)

## Generating
Need to download the vgg19 model and put it with main.py

```
python main.py
```
This command will generate the texture with only the style loss, without the long-range loss in the paper.
This code will generate the texture with a noisy image.

```
python main.py --cc_loss
```
This command will generate the texture with the style loss and the long-range loss in the paper.

```
python main.py --content --content_path ./images/content.jpg --style_path ./images/brick2.jpg --iteration 3000 --cc_loss
```
This command will use the content imageA and style imageB, which means generate images with the content in imageA and style in imageB.
This command will not use the long-range loss in the paper.

```
python main.py --content --content_path ./images/content.jpg --style_path ./images/brick2.jpg --iteration 3000
```
This command has same function about the last command, but it will use the long-range loss.

## Examples

<p>
Style: <br/>
<img src="https://github.com/ty625911724/texture_generation_master/blob/master/images/brick.jpg?raw=true" width="50%"/> <br/>
Output without cc_loss: <br/>
<img src="https://github.com/ckmarkoh/neuralart_tensorflow/blob/master/images/StarryNight.jpg?raw=true" width="50%"/> <br/>
Output: <br/>
<img src="https://github.com/ckmarkoh/neuralart_tensorflow/blob/master/images/Taipei101_StarryNight.jpg?raw=true" width="50%"/> <br/>
</p>
Without the cc_loss, the generated texture is mixed and disorderly.
With the help of the long-range loss, the rendering of regular textures is better.

It has the same effect when combine the content image and style image.

You could change the hyperparameters to get better peformance.
