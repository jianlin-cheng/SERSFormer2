# SERSFormer-2.0 : Leveraging SERS and transformer models for simultaneous detection of multiple pesticides in fresh produce
![SERSFormer2](./blockdiag2_0.png?raw=true "SERSFormer2.0 Architecture Diagram")

The SERSFormer-2.0, is a multi-tasking weight sharing transformer based model, designed for identifying and quantifying the pesticide residue present on the foodsample. It takes SERS spectrum of food sample as input and performs two tasks- classification and regression simultaneously. The above block diagram shows the multi-tasking architecture of SERSFormer. 

The repository contains the SERS dataspectra for 5 different pesticides that are commonly found on spinach, thiabendazole, phosmet, coumaphos, carbophenothion and oxamyl respectiveely and a control sam[le without any pesticides. Each pesticide contains, 5 different concentration ranges from 0 tp 10 ppm.]

To use this repository, clone the repository to required folder on your system using 

`git clone https://github.com/BioinfoMachineLearning/SERSFormer.git`

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

Hajikhani, M., Hegde, A., Snyder, J., Cheng, J., & Lin, M. (2024). Integrating transformer-based machine learning with SERS technology for the analysis of hazardous pesticides in spinach. Journal of Hazardous Materials, 134208. https://doi.org/10.1016/j.jhazmat.2024.134208



