# TrajSurv
Code base for the MLHC 2025 [paper](https://proceedings.mlr.press/v298/zeng25a.html)

## Abstract
Trustworthy survival prediction is essential for clinical decision making. Longitudinal electronic health records (EHRs) provide a uniquely powerful opportunity for the prediction. However, it is challenging to accurately model the continuous clinical progression of patients underlying the irregularly sampled clinical features and to transparently link the progression to survival outcomes. To address these challenges, we develop TrajSurv, a model that learns continuous latent trajectories from longitudinal EHR data for trustworthy survival prediction. TrajSurv employs a neural controlled differential equation (NCDE) to extract continuous-time latent states from the irregularly sampled data, forming continuous latent trajectories. To ensure the latent trajectories reflect the clinical progression, TrajSurv aligns the latent state space with patient state space through a time-aware contrastive learning approach. To transparently link clinical progression to the survival outcome, TrajSurv uses latent trajectories in a two-step divide-and-conquer interpretation process. First, it explains how the changes in clinical features translate into the latent trajectory's evolution using a learned vector field. Second, it clusters these latent trajectories to identify key clinical progression patterns associated with different survival outcomes. Evaluations on two real-world medical datasets, MIMIC-III and eICU, show TrajSurv's competitive accuracy and superior transparency over existing deep learning methods.

## Code
See the [code](./ncde) for the implementation of TrajSurv.

## Citation
If you find TrajSurv useful in your research, please cite our paper:
```bibtex
@InProceedings{pmlr-v298-zeng25a,
  title = 	 {TrajSurv: Learning Continuous Latent Trajectories from Electronic Health Records for Trustworthy Survival Prediction},
  author =       {Zeng, Sihang and Liu, Lucas Jing and Wen, Jun and Yetisgen, Meliha and Etzioni, Ruth and Luo, Gang},
  booktitle = 	 {Proceedings of the 10th Machine Learning for Healthcare Conference},
  year = 	 {2025},
  editor = 	 {Agrawal, Monica and Deshpande, Kaivalya and Engelhard, Matthew and Joshi, Shalmali and Tang, Shengpu and Urteaga, Iñigo},
  volume = 	 {298},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {15--16 Aug},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v298/main/assets/zeng25a/zeng25a.pdf},
  url = 	 {https://proceedings.mlr.press/v298/zeng25a.html},
  abstract = 	 {Trustworthy survival prediction is essential for clinical decision making. Longitudinal electronic health records (EHRs) provide a uniquely powerful opportunity for the prediction. However, it is challenging to accurately model the continuous clinical progression of patients underlying the irregularly sampled clinical features and to transparently link the progression to survival outcomes. To address these challenges, we develop TrajSurv, a model that learns continuous latent trajectories from longitudinal EHR data for trustworthy survival prediction. TrajSurv employs a neural controlled differential equation (NCDE) to extract continuous-time latent states from the irregularly sampled data, forming continuous latent trajectories. To ensure the latent trajectories reflect the clinical progression, TrajSurv aligns the latent state space with patient state space through a time-aware contrastive learning approach. To transparently link clinical progression to the survival outcome, TrajSurv uses latent trajectories in a two-step divide-and-conquer interpretation process. First, it explains how the changes in clinical features translate into the latent trajectory’s evolution using a learned vector field. Second, it clusters these latent trajectories to identify key clinical progression patterns associated with different survival outcomes. Evaluations on two real-world medical datasets, MIMIC-III and eICU, show TrajSurv’s competitive accuracy and superior transparency over existing deep learning methods.}
}
```