# SERSFormer-2.0 : Leveraging SERS and transformer models for simultaneous detection of multiple pesticides in fresh produce
![SERSFormer2](./blockdiag2_0.png?raw=true "SERSFormer2.0 Architecture Diagram")

We introduce SERSFormer 2.0, an advanced multi-task and multi-attention-based transformer model designed for precise detection and quantification of mixed pesticides in agricultural products like spinach and strawberries. It takes SERS spectrum of food sample as input and performs two tasks- multi label classification and multi regression simultaneously. The above block diagram shows the multi-tasking architecture of SERSFormer2.0. 

The repository contains the SERS dataspectra samples for 10 different combination mixtures of pesticide that are commonly found on spinach and strawberry, thiabendazole, phosmet, coumaphos, carbophenothion and oxamyl respectiveely and a control sample without any pesticides. Each pesticide mixture contains, 5 different concentration ranges from 0 tp 10 ppm.

To use this repository, clone the repository to required folder on your system using 

`[git clone https://github.com/BioinfoMachineLearning/SERSFormer.git](https://github.com/jianlin-cheng/SERSFormer2.git)`

set up conda environement and install necessary packages using the setup.sh script.

```
cd SERSFormer2
./setup.sh 
```
To train the model, validate and test, run the following command:
```
python SERSFormer_Training.py \
--attn_head 4 \
--encoder_layers 4\
--save_dir SERSFormer_log\
--entity_name YourWandbUserName 
```
SERSFormer2 uses Wandb for logging all the metrics and training parameters. Provide wandb login username in the arguement to monitor training in realtime. It can be customized to log any media, text, images, graphs, gradients, and metrics. For more information on setting up wandb, please visit the documentation https://docs.wandb.ai/guides/integrations/lightning

**Cite Us**

If this repository is useful, please cite us.

Hegde, A., Hajikhani, M., Snyder, J., Cheng, J., & Lin, M. (2024). Leveraging SERS and transformer models for simultaneous detection of multiple pesticides in fresh produce.

