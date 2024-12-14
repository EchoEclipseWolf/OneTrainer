# Optimizers

This document lists the optimizers available in OneTrainer.

## BNB Standard & 8-bit

* **ADAGRAD**: A deterministic subgradient method with per-parameter learning rates.  Also available in 8-bit precision (**ADAGRAD_8BIT**).

| Setting | Default | Description |
|---|---|---|
| `lr_decay` | 0 | Learning rate decay. |
| `weight_decay` | 0 | Weight decay (L2 penalty). |
| `initial_accumulator_value` | 0 | Initial value for the accumulator. |
| `eps` | 1e-10 | Term added to the denominator to improve numerical stability. |
| `optim_bits` | 32 | Optimizer precision in bits. |
| `min_8bit_size` | 4096 | Minimum size for 8-bit optimization. |
| `percentile_clipping` | 100 | Percentile clipping. |
| `block_wise` | True | Whether to use block-wise optimization. |

* **ADAM**: A popular adaptive optimization algorithm. Also available in 8-bit precision (**ADAM_8BIT**).  The 32-bit version is provided by PyTorch.  See the dedicated **ADAM** and **ADAMW** sections for settings.

| Setting | Default | Description |
|---|---|---|
| `beta1` | 0.9 | Coefficient used for computing running averages of gradient. |
| `beta2` | 0.999 | Coefficient used for computing running averages of squared gradient. |
| `eps` | 1e-8 | Term added to the denominator to improve numerical stability. |
| `weight_decay` | 0 | Weight decay (L2 penalty). |
| `amsgrad` | False | Whether to use the AMSGrad variant of this algorithm. |
| `optim_bits` | 32 | Optimizer precision in bits. |
| `min_8bit_size` | 4096 | Minimum size for 8-bit optimization. |
| `percentile_clipping` | 100 | Percentile clipping. |
| `block_wise` | True | Whether to use block-wise optimization. |
| `is_paged` | False | Whether to use paged optimization. |


* **AdEMAMix**: Adaptive Exponential Moving Average Mix. Also available in 8-bit precision (**AdEMAMix_8BIT**).

| Setting | Default | Description |
|---|---|---|
| `beta1` | 0.9 | Coefficient used for computing running averages of gradient. |
| `beta2` | 0.999 | Coefficient used for computing running averages of squared gradient. |
| `beta3` | 0.9999 | Coefficient used for computing running averages of gradient instability. |
| `eps` | 1e-8 | Term added to the denominator to improve numerical stability. |
| `alpha` | 5 | Coefficient for mixing exponential moving averages. |
| `weight_decay` | 1e-2 | Weight decay (L2 penalty). |
| `min_8bit_size` | 4096 | Minimum size for 8-bit optimization. |
| `is_paged` | False | Whether to use paged optimization. |


* **LAMB**: Layer-wise Adaptive Moments optimizer for Batch training. Also available in 8-bit precision (**LAMB_8BIT**).

| Setting | Default | Description |
|---|---|---|
| `bias_correction` | True | Whether to apply bias correction. |
| `beta1` | 0.9 | Coefficient used for computing running averages of gradient. |
| `beta2` | 0.999 | Coefficient used for computing running averages of squared gradient. |
| `eps` | 1e-8 | Term added to the denominator to improve numerical stability. |
| `weight_decay` | 0 | Weight decay (L2 penalty). |
| `amsgrad` | False | Whether to use the AMSGrad variant of this algorithm. |
| `adam_w_mode` | True | Whether to use AdamW mode. |
| `optim_bits` | 32 | Optimizer precision in bits. |
| `min_8bit_size` | 4096 | Minimum size for 8-bit optimization. |
| `percentile_clipping` | 100 | Percentile clipping. |
| `block_wise` | False | Whether to use block-wise optimization. |
| `max_unorm` | 1.0 | Maximum unorm. |

* **LARS**: Layer-wise Adaptive Rate Scaling. Also available in 8-bit precision (**LARS_8BIT**).

| Setting | Default | Description |
|---|---|---|
| `momentum` | 0 | Momentum factor. |
| `dampening` | 0 | Dampening for momentum. |
| `weight_decay` | 0 | Weight decay (L2 penalty). |
| `nesterov` | False | Whether to use Nesterov momentum. |
| `optim_bits` | 32 | Optimizer precision in bits. |
| `min_8bit_size` | 4096 | Minimum size for 8-bit optimization. |
| `percentile_clipping` | 100 | Percentile clipping. |
| `max_unorm` | 0.02 | Maximum unorm. |


* **LION**: Lion optimizer. Also available in 8-bit precision (**LION_8BIT**).  See dedicated **LION** section for settings.

| Setting | Default | Description |
|---|---|---|
| `beta1` | 0.9 | Coefficient used for computing running averages of gradient. |
| `beta2` | 0.999 | Coefficient used for computing running averages of squared gradient. |
| `weight_decay` | 0 | Weight decay (L2 penalty). |
| `min_8bit_size` | 4096 | Minimum size for 8-bit optimization. |
| `percentile_clipping` | 100 | Percentile clipping. |
| `block_wise` | True | Whether to use block-wise optimization. |
| `is_paged` | False | Whether to use paged optimization. |


* **RMSPROP**: Root Mean Square Propagation. Also available in 8-bit precision (**RMSPROP_8BIT**).

| Setting | Default | Description |
|---|---|---|
| `alpha` | 0.99 | Smoothing constant. |
| `eps` | 1e-8 | Term added to the denominator to improve numerical stability. |
| `weight_decay` | 0 | Weight decay (L2 penalty). |
| `momentum` | 0 | Momentum factor. |
| `centered` | False | Whether to use the centered variant of RMSprop. |
| `optim_bits` | 32 | Optimizer precision in bits. |
| `min_8bit_size` | 4096 | Minimum size for 8-bit optimization. |
| `percentile_clipping` | 100 | Percentile clipping. |
| `block_wise` | True | Whether to use block-wise optimization. |

* **SGD**: Stochastic Gradient Descent. Also available in 8-bit precision (**SGD_8BIT**). The 32-bit version is provided by PyTorch. See dedicated **SGD** section for settings.

| Setting | Default | Description |
|---|---|---|
| `momentum` | 0 | Momentum factor. |
| `dampening` | 0 | Dampening for momentum. |
| `weight_decay` | 0 | Weight decay (L2 penalty). |
| `nesterov` | False | Whether to use Nesterov momentum. |
| `min_8bit_size` | 4096 | Minimum size for 8-bit optimization. |
| `percentile_clipping` | 100 | Percentile clipping. |
| `block_wise` | True | Whether to use block-wise optimization. |


## Schedule-free optimizers


* **SCHEDULE_FREE_ADAMW**: A variant of ADAMW that doesn't require a learning rate schedule.
* **SCHEDULE_FREE_SGD**: A variant of SGD that doesn't require a learning rate schedule.

## DADAPT

These optimizers are adaptive.

* **DADAPT_ADA_GRAD**:  DADAPT variant of ADAGRAD.
* **DADAPT_ADAM**: DADAPT variant of ADAM.
* **DADAPT_ADAN**: DADAPT variant of ADAN.
* **DADAPT_LION**: DADAPT variant of LION.
* **DADAPT_SGD**: DADAPT variant of SGD.

## Prodigy

* **PRODIGY**:  An adaptive optimizer.

## CAME

* **CAME**:  Combined Adaptive Momentum Estimation. Supports fused back pass.

## ADAFACTOR

* **ADAFACTOR**:  An adaptive optimizer that saves memory by not storing per-parameter moment estimates.  Supports fused back pass.

| Setting | Default | Description |
|---|---|---|
| `eps` | 1e-30 | Regularization constant for squared gradient. |
| `eps2` | 1e-3 | Regularization constant for root mean square. |
| `clip_threshold` | 1.0 | Threshold of root-mean-square of final gradient update. |
| `decay_rate` | -0.8 | Coefficient used for computing running averages of squared gradient. |
| `beta1` | `None` | Coefficient used for computing running averages of gradient.  If `None`, momentum is not used. |
| `weight_decay` | 0.0 | Weight decay (L2 penalty). |
| `scale_parameter` | `False` |  |
| `relative_step` | `False` |  |
| `warmup_init` | `False` |  |
| `stochastic_rounding` | `True` | Whether to use stochastic rounding. |
| `fused_back_pass` | `False` | Whether to use fused back pass. |


## ADAMW

* **ADAMW**: A variant of ADAM with weight decay. Also available in 8-bit precision (**ADAMW_8BIT**). The 32-bit version is provided by PyTorch.

| Setting | Default | Description |
|---|---|---|
| `beta1` | 0.9 | Coefficient used for computing running averages of gradient. |
| `beta2` | 0.999 | Coefficient used for computing running averages of squared gradient. |
| `eps` | 1e-8 | Term added to the denominator to improve numerical stability. |
| `weight_decay` | 1e-2 | Weight decay (L2 penalty). |
| `amsgrad` | `False` | Whether to use the AMSGrad variant of this algorithm. |
| `foreach` | `False` |  |
| `maximize` | `False` |  |
| `capturable` | `False` |  |
| `differentiable` | `False` |  |
| `fused` | `True` |  |
| `stochastic_rounding` | `False` | Whether to use stochastic rounding. |
| `fused_back_pass` | `False` | Whether to use fused back pass. |



## Schedule-free optimizers

* **SCHEDULE_FREE_ADAMW**: A variant of ADAMW that doesn't require a learning rate schedule.

| Setting | Default | Description |
|---|---|---|
| `beta1` | 0.9 | Coefficient used for computing running averages of gradient. |
| `beta2` | 0.999 | Coefficient used for computing running averages of squared gradient. |
| `eps` | 1e-8 | Term added to the denominator to improve numerical stability. |
| `weight_decay` | 1e-2 | Weight decay (L2 penalty). |
| `r` | 0.0 |  |
| `weight_lr_power` | 2.0 |  |
| `foreach` | False | Whether to use foreach optimization. |

* **SCHEDULE_FREE_SGD**: A variant of SGD that doesn't require a learning rate schedule.

| Setting | Default | Description |
|---|---|---|
| `momentum` | 0 | Momentum factor. |
| `weight_decay` | 1e-2 | Weight decay (L2 penalty). |
| `r` | 0.0 |  |
| `weight_lr_power` | 2.0 |  |
| `foreach` | False | Whether to use foreach optimization. |


## DADAPT

These optimizers are adaptive.

* **DADAPT_ADA_GRAD**:  DADAPT variant of ADAGRAD.

| Setting | Default | Description |
|---|---|---|
| `momentum` | 0 | Momentum factor. |
| `log_every` | 0 |  |
| `weight_decay` | 0.0 | Weight decay (L2 penalty). |
| `eps` | 0.0 | Term added to the denominator to improve numerical stability. |
| `d0` | 1e-6 |  |
| `growth_rate` | inf |  |

* **DADAPT_ADAM**: DADAPT variant of ADAM.

| Setting | Default | Description |
|---|---|---|
| `beta1` | 0.9 | Coefficient used for computing running averages of gradient. |
| `beta2` | 0.999 | Coefficient used for computing running averages of squared gradient. |
| `eps` | 1e-8 | Term added to the denominator to improve numerical stability. |
| `weight_decay` | 0 | Weight decay (L2 penalty). |
| `log_every` | 0 |  |
| `decouple` | False |  |
| `use_bias_correction` | False |  |
| `d0` | 1e-6 |  |
| `growth_rate` | inf |  |
| `fsdp_in_use` | False |  |

* **DADAPT_ADAN**: DADAPT variant of ADAN.

| Setting | Default | Description |
|---|---|---|
| `beta1` | 0.98 | Coefficient used for computing running averages of gradient. |
| `beta2` | 0.92 | Coefficient used for computing running averages of squared gradient. |
| `beta3` | 0.99 | Coefficient used for computing running averages of gradient instability. |
| `eps` | 1e-8 | Term added to the denominator to improve numerical stability. |
| `weight_decay` | 0.02 | Weight decay (L2 penalty). |
| `no_prox` | False |  |
| `log_every` | 0 |  |
| `d0` | 1e-6 |  |
| `growth_rate` | inf |  |

* **DADAPT_LION**: DADAPT variant of LION.

| Setting | Default | Description |
|---|---|---|
| `beta1` | 0.9 | Coefficient used for computing running averages of gradient. |
| `beta2` | 0.999 | Coefficient used for computing running averages of squared gradient. |
| `weight_decay` | 0.0 | Weight decay (L2 penalty). |
| `log_every` | 0 |  |
| `d0` | 1e-6 |  |
| `fsdp_in_use` | False |  |

* **DADAPT_SGD**: DADAPT variant of SGD.

| Setting | Default | Description |
|---|---|---|
| `momentum` | 0.0 | Momentum factor. |
| `weight_decay` | 0 | Weight decay (L2 penalty). |
| `log_every` | 0 |  |
| `d0` | 1e-6 |  |
| `growth_rate` | inf |  |
| `fsdp_in_use` | False |  |


## Prodigy

* **PRODIGY**:  An adaptive optimizer.

| Setting | Default | Description |
|---|---|---|
| `beta1` | 0.9 | Coefficient used for computing running averages of gradient. |
| `beta2` | 0.999 | Coefficient used for computing running averages of squared gradient. |
| `beta3` | None | Coefficient used for computing running averages of gradient instability. |
| `eps` | 1e-8 | Term added to the denominator to improve numerical stability. |
| `weight_decay` | 0 | Weight decay (L2 penalty). |
| `decouple` | True |  |
| `use_bias_correction` | False |  |
| `safeguard_warmup` | False |  |
| `d0` | 1e-6 |  |
| `d_coef` | 1.0 |  |
| `growth_rate` | inf |  |
| `fsdp_in_use` | False |  |


## PyTorch Optimizers

* **ADAM**: A PyTorch implementation of Adam. See the **ADAMW** section for similar settings, noting that `ADAM` defaults `weight_decay` to 0 and `fused` to True.  Additionally:

| Setting | Default | Description |
|---|---|---|
| `stochastic_rounding` | False | Whether to use stochastic rounding. |
| `fused_back_pass` | False | Whether to use fused back pass. |

* **SGD**: A PyTorch implementation of Stochastic Gradient Descent.

| Setting | Default | Description |
|---|---|---|
| `momentum` | 0 | Momentum factor. |
| `dampening` | 0 | Dampening for momentum. |
| `weight_decay` | 0 | Weight decay (L2 penalty). |
| `nesterov` | False | Whether to use Nesterov momentum. |
| `foreach` | False | Whether to use foreach optimization. |
| `maximize` | False | Whether to maximize instead of minimize the objective function. |
| `differentiable` | False | Whether to compute gradients with respect to the optimizer's state. |

* **LION**: A PyTorch implementation of Lion.

| Setting | Default | Description |
|---|---|---|
| `beta1` | 0.9 | Coefficient used for computing running averages of gradient. |
| `beta2` | 0.99 | Coefficient used for computing running averages of squared gradient. |
| `weight_decay` | 0.0 | Weight decay (L2 penalty). |
| `use_triton` | False | Whether to use triton for acceleration. |


* **ADABELIEF**: A PyTorch optimizer that combines Adam and RMSprop.

| Setting | Default | Description |
|---|---|---|
| `beta1` | 0.9 | Coefficient used for computing running averages of gradient. |
| `beta2` | 0.999 | Coefficient used for computing running averages of squared gradient. |
| `eps` | 1e-16 | Term added to the denominator to improve numerical stability. |
| `weight_decay` | 0 | Weight decay (L2 penalty). |
| `amsgrad` | False | Whether to use the AMSGrad variant. |
| `decoupled_decay` | True | Whether to use decoupled weight decay. |
| `fixed_decay` | False | Whether to use fixed weight decay. |
| `rectify` | True | Whether to use the rectified Adam update. |
| `degenerated_to_sgd` | True | Whether to degenerate to SGD. |

* **TIGER**: A PyTorch optimizer.

| Setting | Default | Description |
|---|---|---|
| `beta1` | 0.965 | Coefficient used for computing running averages of gradient. |
| `weight_decay` | 0.01 | Weight decay (L2 penalty). |
| `decoupled_decay` | True | Whether to use decoupled weight decay. |
| `fixed_decay` | False | Whether to use fixed weight decay. |

* **AIDA**: A PyTorch optimizer.

| Setting | Default | Description |
|---|---|---|
| `beta1` | 0.9 | Coefficient used for computing running averages of gradient. |
| `beta2` | 0.999 | Coefficient used for computing running averages of squared gradient. |
| `k` | 2 |  |
| `xi` | 1e-20 |  |
| `weight_decay` | 0.0 | Weight decay (L2 penalty). |
| `decoupled_decay` | False | Whether to use decoupled weight decay. |
| `fixed_decay` | False | Whether to use fixed weight decay. |
| `rectify` | False | Whether to use the rectified Adam update. |
| `n_sma_threshold` | 5 |  |
| `degenerated_to_sgd` | True | Whether to degenerate to SGD. |
| `ams_bound` | False | Whether to use AMS bound. |
| `r` | 0.95 |  |
| `adanorm` | False | Whether to use AdaNorm. |
| `adam_debias` | False | Whether to use Adam debias. |
| `eps` | 1e-8 | Term added to the denominator to improve numerical stability. |




* **ADABELIEF**: A PyTorch optimizer that combines Adam and RMSprop.
* **TIGER**: A PyTorch optimizer.
* **AIDA**: A PyTorch optimizer.
