# texture_generation_master

Implementation of paper: Incorporating long-range consistency in CNN-based texture generation (https://arxiv.org/pdf/1606.01286.pdf) by Tensorflow.

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
This command will generate the texture with only the style loss, without the long-range loss in the paper.<br/>
This code will generate the texture with a noisy image.

```
python main.py --cc_loss
```
This command will generate the texture with the style loss and the long-range loss in the paper.

```
python main.py --content --content_path ./images/content.jpg --style_path ./images/brick2.jpg --iteration 3000 --cc_loss
```
This command will use the content imageA and style imageB, which means generate images with the content in imageA and style in imageB.<br/>
This command will not use the long-range loss in the paper.

```
python main.py --content --content_path ./images/content.jpg --style_path ./images/brick2.jpg --iteration 3000
```
This command has same function about the last command, but it will use the long-range loss.

## Examples
<p>
<img src="https://github.com/ty625911724/texture_generation_master/blob/master/images/brick.jpg?raw=true" width="48%" title="style image"/>
<img src="https://github.com/ty625911724/texture_generation_master/blob/master/images/noisy.jpg?raw=true" width="48%"/><br/>

**The left is style image,the right is content Input.**<br/>
<img src="https://github.com/ty625911724/texture_generation_master/blob/master/images/results_style.png?raw=true" width="48%"/>
<img src="https://github.com/ty625911724/texture_generation_master/blob/master/images/results_cc.png?raw=true" width="48%"/><br/>

**The left is Output without cc_loss, the right is the Output with cc_loss.** <br/>
**Without the cc_loss, the generated texture is mixed and disorderly.**<br/>
**With the help of the long-range loss, the rendering of regular textures is better**.<br/>

<p>
<img src="https://github.com/ty625911724/texture_generation_master/blob/master/images/brick2.jpg?raw=true" width="48%" title="style image"/>
<img src="https://github.com/ty625911724/texture_generation_master/blob/master/images/content.jpg?raw=true" width="48%"/><br/>

**The left is style image,the right is content Input.**<br/>

<img src="https://github.com/ty625911724/texture_generation_master/blob/master/images/results_style_car.png?raw=true" width="48%"/><img src="https://github.com/ty625911724/texture_generation_master/blob/master/images/results_cc_car.png?raw=true" width="48%"/><br/>

**The left is Output without cc_loss, the right is the Output with cc_loss.** <br/>
**The regular texture is also generated better when combining the content image and style image.**<br/>

**You could change the hyperparameters to get better peformance.**
