# Text-to-Image Translation (DALL-E) for TPU in Pytorch

Refactoring 
[Taming Transformers](https://github.com/CompVis/taming-transformers) and [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch)
for TPU VM with [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

## Requirements

```
pip install -r requirements.txt
```

## Data Preparation

Place any image dataset with ImageNet-style directory structure (at least 1 subfolder) to fit the dataset into pytorch ImageFolder.

## Training VQVAEs
You can easily test main.py with randomly generated fake data.
```
python train_vae.py --use_tpus --fake_data
```

For actual training provide specific directory for train_dir, val_dir, log_dir:

```
python train_vae.py --use_tpus --train_dir [training_set] --val_dir [val_set] --log_dir [where to save results]
```

## Training DALL-E
```
python train_dalle.py --use_tpus --train_dir [training_set] --val_dir [val_set] --log_dir [where to save results] --vae_path [pretrained vae] --bpe_path [pretrained bpe(optional)]
```

## TODO
- [x] Add VQVAE, VQGAN, and Gumbel VQVAE(Discrete VAE), Gumbel VQGAN
- [x] Add [VQVAE2](https://arxiv.org/abs/1906.00446)
- [x] Add EMA update for Vector Quantization
- [x] Debug VAEs (Single TPU Node, TPU Pods, GPUs)
- [x] Resolve SIGSEGV issue with large TPU Pods [pytorch-xla #3028](https://github.com/pytorch/xla/issues/3028)
- [x] Add DALL-E
- [x] Debug DALL-E (Single TPU Node, TPU Pods, GPUs)
- [x] Add WebDataset support
- [ ] Debug WebDataset functionality
- [ ] Add Image Logger by modifying pl_bolts TensorboardGenerativeModelImageSampler()
- [ ] Add automatic checkpoint saver and resume for sudden (which happens a lot) TPU restart
- [ ] Add [RBGumbelQuantizer](https://arxiv.org/abs/2010.04838)
- [ ] Add [HiT](https://arxiv.org/abs/2106.07631)

## BibTeX

```
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```
@misc{ramesh2021zeroshot,
    title   = {Zero-Shot Text-to-Image Generation}, 
    author  = {Aditya Ramesh and Mikhail Pavlov and Gabriel Goh and Scott Gray and Chelsea Voss and Alec Radford and Mark Chen and Ilya Sutskever},
    year    = {2021},
    eprint  = {2102.12092},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```


